import asyncio
import numpy as np
from pathlib import Path
from .pathfinder import astar, astar_with_hazard, find_frontier, find_safe_frontier
from .local_risk_map import LocalRiskMap
from .global_visualizer import AsyncGlobalMap
from .rl_policy import DQNAgent
from .perception_classifier import PerceptionClassifier
from .risk_scorer import RiskScorer
import os
import csv
import time
import math

class AgentTask:
    def __init__(self, agent_id, handler_service):
        self.agent_id = agent_id
        self.handler = handler_service

        self.inbox = asyncio.Queue()

        # FIX: Match sandbox resolution (1.0) so state vector normalization aligns
        self.local_map = LocalRiskMap(grid_size=30, resolution=1)
        
        self.assigned_zone = []
        self.current_pos = (0, 0)
        self.path = []
        
        # FIX: Track continuous threat exposure exactly like the sandbox
        self.threat_history = False
        self.last_risk_by_threat = {}

        self.perception = PerceptionClassifier()
        self.risk_scorer = RiskScorer()
        self.policy = DQNAgent(state_dim=6, action_dim=4) 
        
        checkpoint_path = Path(__file__).parent / "checkpoints" / "framework_v1_final.pt"

        self.log_file = "python_telemetry.csv"
        # Write headers only if the file doesn't exist yet
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "agent_id", "event_trigger", "rl_action_idx", 
                    "risk_scalar", "threat_history", "grid_x", "grid_z", "path_length"
                ])

        if checkpoint_path.exists():
            self.policy.load(checkpoint_path)
        else:
            print(f"Warning: no checkpoint found at {checkpoint_path}, agent {agent_id} running with untrained weights")

        asyncio.create_task(self.run())

    async def run(self):
        print(f"Agent {self.agent_id} running...")

        while True:
            event = await self.inbox.get()
            event_type = event.get("type")

            if event_type == "zone_update":
                await self._handle_zone_update(event["cells"])

            elif event_type == "waypoint_reached":
                await self._handle_waypoint_reached(event["payload"]["pos"])

            elif event_type == "obstacle_blocked":
                await self._handle_obstacle_blocked(event["payload"]["obstacle_pos"], event["payload"].get("size"))

            elif event_type == "sensor_detection":
                await self._handle_sensor_detection(event["payload"]["tags"], event["payload"]["distance"], event["payload"]["hazard_pos"])

            elif event_type == "agent_stuck":
                await self._handle_agent_stuck(event["payload"]["pos"])

            self.inbox.task_done()

    async def _handle_agent_stuck(self, pos):
        # If there is no path, just force a replan without dropping blind obstacles.
        if not self.path:
            self._log_python_telemetry(event_trigger="agent_stuck_nopath")
            await self._replan_standard()
            return

        # The agent is stuck trying to physically reach the NEXT waypoint.
        # We mark that specific target cell as the impassable blockage, not the agent's current cell.
        next_wp = self.path[0]
        
        # radius=1 creates a 3x3 blocked area around the waypoint, 
        # which is plenty to force A* to route around it.
        self.local_map.mark_impassable(next_wp, radius=1)

        print(f"[TELEMETRY] Agent {self.agent_id} stuck. Marked target waypoint {next_wp} as impassable.")
        
        # Clear the doomed path and ask A* for a new route
        self.path = []
        self._log_python_telemetry(event_trigger="agent_stuck")
        await self._replan_standard()

    async def _handle_zone_update(self, cells):
        # FIX: Handle JSON dicts {"x": 1, "z": 2} from WebSocket
        parsed_cells = [(c['x'], c['z']) if isinstance(c, dict) else c for c in cells]
        self.assigned_zone = parsed_cells
        self.local_map.update_zone_mask(parsed_cells)

        # Sync the new Voronoi partition to Unity for rendering
        formatted_cells = [{"x": cx, "z": cz} for cx, cz in parsed_cells]
        print(f"[DEBUG] Agent {self.agent_id} firing voronoi_sync with {len(formatted_cells)} cells.")
        await self._send_to_unity("voronoi_sync", {"cells": formatted_cells})

        await self._replan_standard()

    async def _handle_waypoint_reached(self, current_pos):
        # Extract the raw coordinates regardless of dict or list format
        raw_x = current_pos['x'] if isinstance(current_pos, dict) else current_pos[0]
        raw_z = current_pos['z'] if isinstance(current_pos, dict) else current_pos[1]
        
        # 1. Translate Unity World -> Python Grid AND force it to be a tuple
        grid_pos = self.local_map.world_to_grid(raw_x, raw_z)
        self.current_pos = tuple(grid_pos) 
        self.local_map.mark_explored(self.current_pos)
        if hasattr(self.handler, 'global_map'):
            self.handler.global_map.update_rover_position(
                self.agent_id, 
                raw_x, 
                raw_z
            )
        await self._send_to_unity("occupancy_update", {
            "x": self.current_pos[0],
            "z": self.current_pos[1],
            "state": 1
        })

        if self.path:
            next_wp = self.path.pop(0)
            world_x, world_z = self.local_map.grid_to_world(next_wp[0], next_wp[1])
            await self._send_to_unity("waypoint_list", [{"x": world_x, "z": world_z}])
        else:
            await self._replan_standard()

    async def _handle_obstacle_blocked(self, obstacle_pos, size=None):
        raw_x = obstacle_pos['x'] if isinstance(obstacle_pos, dict) else obstacle_pos[0]
        raw_z = obstacle_pos['z'] if isinstance(obstacle_pos, dict) else obstacle_pos[1]
        grid_pos = self.local_map.world_to_grid(raw_x, raw_z)

        radius = 1
        if size:
            radius = max(1, int(np.ceil(max(size['x'] if isinstance(size, dict) else size[0],
                                            size['z'] if isinstance(size, dict) else size[1]) / self.local_map.resolution / 2)))

        already_blocked = self.local_map.grid[grid_pos[1], grid_pos[0]] == 1
        self.local_map.mark_impassable(grid_pos, radius=radius)
        
        #Push the new impassable obstacle to the global map
        if hasattr(self.handler, 'global_map'):
            self.handler.global_map.merge_local_state(self.local_map.explored, self.local_map.grid)

        if already_blocked and self.path:
            return

        self.path = []
        self._log_python_telemetry(event_trigger="obstacle_hit")
        await self._replan_standard()

    async def _handle_sensor_detection(self, tag, distance, hazard_pos):
        raw_x = hazard_pos['x'] if isinstance(hazard_pos, dict) else hazard_pos[0]
        raw_z = hazard_pos['z'] if isinstance(hazard_pos, dict) else hazard_pos[1]
        
        grid_hazard_pos = self.local_map.world_to_grid(raw_x, raw_z)

        feature_vector = self.perception.classify(tag)
        risk_scalar = self.risk_scorer.calculate_immediate_risk(feature_vector, distance)
        print(f"Lethality(in handle detection): {feature_vector} tag: {tag} distance: {distance} risk_scalar: {risk_scalar}")

        if hasattr(self.handler, 'global_map') and feature_vector:
            radius_cells = int(math.ceil(feature_vector["radius"] / self.local_map.resolution))
            self.handler.global_map.merge_local_state(
                self.local_map.explored, 
                self.local_map.grid, 
                hazard_pos=grid_hazard_pos, 
                hazard_radius=radius_cells,  
                tag=tag,
            )

        key = grid_hazard_pos
        prev_risk = self.last_risk_by_threat.get(key, 0.0)

        if risk_scalar < 0.01:
            self.last_risk_by_threat.pop(key, None)
        elif risk_scalar > 0.02 and risk_scalar > prev_risk:
            self.threat_history = True
            await self._execute_rl_decision(risk_scalar, feature_vector, grid_hazard_pos)

        self.last_risk_by_threat[key] = risk_scalar

    async def _execute_rl_decision(self, risk_scalar, feature_vector, hazard_pos):
        state_vector = np.array([
            self.current_pos[0] / self.local_map.grid_width,
            self.current_pos[1] / self.local_map.grid_height,
            np.tanh(risk_scalar),
            self.local_map.get_coverage_pct(),
            1.0, # alive
            float(self.threat_history) # FIX: Dynamic threat history
        ], dtype=np.float32)

        action_idx = int(self.policy.act(state_vector, epsilon=0.0))

        self._log_python_telemetry(event_trigger="rl_decision", action_idx=action_idx, risk_scalar=risk_scalar)

        # FIX: Action Space is exactly 4 dimensions (HOLD removed)
        if action_idx == 0: # CONTINUE
            self.threat_history = False
            
        elif action_idx == 1: # REROUTE
            self.threat_history = False

            hazard_cost_map = self.local_map.build_hazard_cost_map(hazard_pos, feature_vector)
            active_threats = [{'pos': (hazard_pos[0], hazard_pos[1]), 'features': feature_vector}]
            target = find_safe_frontier(
                self.local_map.explored, 
                self.local_map.zone_mask,
                self.local_map.grid, 
                self.current_pos, 
                lambda cx, cz: self.risk_scorer.calculate_cell_risk(cx, cz, active_threats)
            )
            if target:
                new_path = astar_with_hazard(self.local_map.grid, hazard_cost_map, self.current_pos, target)
                self.path = new_path[1:] if new_path else []
                # Convert path to Unity's format
                formatted_path = []
                for wp in self.path:
                    world_x, world_z = self.local_map.grid_to_world(wp[0], wp[1])
                    formatted_path.append({"x": world_x, "z": world_z})
                
                print(f"[TELEMETRY] RL Action 1 sending path: {formatted_path}")
                await self._send_to_unity("waypoint_list", formatted_path)

        elif action_idx == 2: # MARK_DANGER
            self.threat_history = False

            await self.handler.comms_client.send_to_coordinator(
                "threat_broadcast",
                self.agent_id,
                {
                    "x": hazard_pos[0],
                    "z": hazard_pos[1], 
                    "risk_scalar": risk_scalar, 
                    "radius_m": feature_vector["radius"]
                }
            )
            # Re-route after marking just like the sandbox
            hazard_cost_map = self.local_map.build_hazard_cost_map(hazard_pos, feature_vector)
            active_threats = [{'pos': (hazard_pos[0], hazard_pos[1]), 'features': feature_vector}]
            target = find_safe_frontier(
                self.local_map.explored, 
                self.local_map.zone_mask,
                self.local_map.grid, 
                self.current_pos, 
                lambda cx, cz: self.risk_scorer.calculate_cell_risk(cx, cz, active_threats)
            )
            if target:
                new_path = astar_with_hazard(self.local_map.grid, hazard_cost_map, self.current_pos, target)
                self.path = new_path[1:] if new_path else []
                formatted_path = []
                for wp in self.path:
                    world_x, world_z = self.local_map.grid_to_world(wp[0], wp[1])
                    formatted_path.append({"x": world_x, "z": world_z})
                
                print(f"[TELEMETRY] RL Action 2 sending path: {formatted_path}")
                await self._send_to_unity("waypoint_list", formatted_path)

        elif action_idx == 3: # REQUEST_REASSIGNMENT
            self.threat_history = False
            await self.handler.comms_client.send_to_coordinator(
                "request_reassignment",
                self.agent_id, 
                # Ensure orphaned cells are serialized as dicts
                {"orphaned_cells": [{"x": c[0], "z": c[1]} for c in self.assigned_zone]}
            )

    async def _replan_standard(self):
        target = find_frontier(
            self.local_map.explored,
            self.local_map.zone_mask,
            self.local_map.grid,
            self.current_pos
        )

        if target:
            full_path = astar(self.local_map.grid, self.current_pos, target)
            self.path = full_path[1:] if full_path else []
            formatted_path = []
            for wp in self.path:
                world_x, world_z = self.local_map.grid_to_world(wp[0], wp[1])
                formatted_path.append({"x": world_x, "z": world_z})

            print(f"[TELEMETRY] Agent {self.agent_id} sending path: {formatted_path}")
            await self._send_to_unity("waypoint_list", formatted_path)

            self._log_python_telemetry(event_trigger="standard_replan")

    async def _send_to_unity(self, msg_type, payload):
        await self.handler.comms_client.send_to_unity(
            msg_type=msg_type,
            agent_id=self.agent_id, 
            payload=payload
        )

    def _log_python_telemetry(self, event_trigger, action_idx="N/A", risk_scalar=0.0):
        """Appends the agent's internal brain state to the CSV and prints live to the console."""
        
        # Live terminal output to track exact execution flow
        print(f"[LIVE TELEMETRY] Agent: {self.agent_id} | Event: {event_trigger} | Pos: {self.current_pos} | Path Len: {len(self.path)} | RL Action: {action_idx}")

        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round(time.time(), 3),
                self.agent_id,
                event_trigger,
                action_idx,
                round(risk_scalar, 4),
                self.threat_history,
                self.current_pos[0],
                self.current_pos[1],
                len(self.path)
            ])