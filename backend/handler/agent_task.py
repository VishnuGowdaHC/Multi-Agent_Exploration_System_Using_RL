import asyncio
import numpy as np
from pathlib import Path
from .pathfinder import astar, astar_with_hazard, find_frontier, find_safe_frontier
from .local_risk_map import LocalRiskMap
from .rl_policy import DQNAgent
from .perception_classifier import PerceptionClassifier
from .risk_scorer import RiskScorer

class AgentTask:
    def __init__(self, agent_id, handler_service):
        self.agent_id = agent_id
        self.handler = handler_service

        self.inbox = asyncio.Queue()

        # FIX: Match sandbox resolution (1.0) so state vector normalization aligns
        self.local_map = LocalRiskMap(grid_size=15, resolution=1.0)
        self.assigned_zone = []
        self.current_pos = (0, 0)
        self.path = []
        
        # FIX: Track continuous threat exposure exactly like the sandbox
        self.threat_history = False
        self.last_risk = 0.0

        self.perception = PerceptionClassifier()
        self.risk_scorer = RiskScorer()
        self.policy = DQNAgent(state_dim=6, action_dim=4) 
        
        checkpoint_path = Path(__file__).parent / "checkpoints" / "framework_v1_final.pt"

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
                await self._handle_waypoint_reached(event["pos"])

            elif event_type == "obstacle_blocked":
                await self._handle_obstacle_blocked(event["obstacle_pos"])

            elif event_type == "sensor_detection":
                await self._handle_sensor_detection(event["tags"], event["distance"], event["hazard_pos"])

            self.inbox.task_done()

    async def _handle_zone_update(self, cells):
        # FIX: Handle JSON dicts {"x": 1, "z": 2} from WebSocket
        parsed_cells = [(c['x'], c['z']) if isinstance(c, dict) else c for c in cells]
        self.assigned_zone = parsed_cells
        self.local_map.update_zone_mask(parsed_cells)

        await self._replan_standard()

    async def _handle_waypoint_reached(self, current_pos):
        # FIX: Parse dict if necessary
        self.current_pos = (current_pos['x'], current_pos['z']) if isinstance(current_pos, dict) else current_pos
        self.local_map.mark_explored(self.current_pos)

        if self.path:
            next_wp = self.path.pop(0)
            # FIX: Convert tuple to dict for Unity JSON serialization
            await self._send_to_unity("waypoint_list", [{"x": next_wp[0], "z": next_wp[1]}])
        else:
            await self._replan_standard()

    async def _handle_obstacle_blocked(self, obstacle_pos):
        obs_pos = (obstacle_pos['x'], obstacle_pos['z']) if isinstance(obstacle_pos, dict) else obstacle_pos
        self.local_map.mark_impassable(obs_pos)
        self.path = []

        await self._replan_standard()

    async def _handle_sensor_detection(self, tag, distance, hazard_pos):
        feature_vector = self.perception.classify(tag)
        risk_scalar = self.risk_scorer.calculate_immediate_risk(feature_vector, distance)

        # FIX: Replicate the sandbox interrupt and "coasting" logic
        if risk_scalar < 0.01:
            self.threat_history = False

        if risk_scalar > 0.02 and risk_scalar > self.last_risk and not self.threat_history:
            self.threat_history = True
            await self._execute_rl_decision(risk_scalar, feature_vector, hazard_pos)
            
        self.last_risk = risk_scalar

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
                formatted_path = [{"x": wp[0], "z": wp[1]} for wp in self.path]
                await self._send_to_unity("waypoint_list", formatted_path)

        elif action_idx == 2: # MARK_DANGER
            self.threat_history = False
            await self.handler.send_to_coordinator(
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
                formatted_path = [{"x": wp[0], "z": wp[1]} for wp in self.path]
                await self._send_to_unity("waypoint_list", formatted_path)

        elif action_idx == 3: # REQUEST_REASSIGNMENT
            self.threat_history = False
            await self.handler.send_to_coordinator(
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
            if self.path:
                formatted_path = [{"x": wp[0], "z": wp[1]} for wp in self.path]
                await self._send_to_unity("waypoint_list", formatted_path)

    async def _send_to_unity(self, msg_type, payload):
        await self.handler.comms_client.send_to_unity(
            msg_type=msg_type,
            agent_id=self.agent_id, 
            payload=payload
        )