import asyncio
import numpy as np

from pathfinding import astar, astar_with_hazard, find_frontier, find_safe_frontier
from local_risk_map import LocalRiskMap
from rl_policy import DQNAgent
from perception_classifier import PerceptionClassifier
from risk_scorer import RiskScorer

class AgentTask:
    def __init__(self, agent_id, handler_service):
        self.agent_id = agent_id
        self.handler_service = handler_service

        self.inbox = asyncio.Queue()

        self.local_map = LocalRiskMap(grid_size=15, resolution=0.5)
        self.assigned_zone = []
        self.current_pos = (0, 0)
        self.path = []
        self.hold_streak = 0

        self.perception = PerceptionClassifier()
        self.risk_scorer = RiskScorer()
        self.policy = DQNAgent()

        self.policy.load()

        asyncio.creat_task(self.run())

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
        self.assigned_zone = cells
        self.local_map.update_zone_mask(cells)

        await self._replan_standard()

    async def _handle_waypoint_reached(self, current_pos):
        self.current_pos = current_pos
        self.local_map.mark_explored(current_pos)

        if self.path:
            next_wp = self.path.pop(0)
            await self._send_to_unity("waypoint_list", [next_wp])

        else:
            await self._replan_standard()

    async def _handle_obstacle_blocked(self, obstacle_pos):
        self.local_map.mark_impassable(obstacle_pos)
        self.path = []

        await self._replan_standard()

    async def _handle_sensor_detection(self, tag, distance, hazard_pas):
        feature_vector = self.perception.classify(tag)

        risk_scalar = self.risk_scorer.calculate(feature_vector, distance)

        if risk_scalar > 0.02:
            await self._execute_rl_decision(risk_scalar, feature_vector, hazard_pas)

    async def _execute_rl_decision(self, risk_scalar, feature_vector, hazard_pos):
        state_vector = np.array([
            self.current_pos[0] / self.local_map.grid_size,
            self.current_pos[1] / self.local_map.grid_size,
            np.tanh(risk_scalar),
            self.local_map.get_coverage_pct(),
            1.0, #alive
            1.0  #threat_history
        ], dtype=np.float32)

        action_idx = self.policy.act(state_vector, epsilon=0.0)

        if action_idx == 0: #CONTINUE
            self.hold_streak = 0

        elif action_idx == 1: #REROUTE
            self.hold_streak = 0
            hazard_cost_map = self.local_map.build_hazard_cost_map(hazard_pos, feature_vector)
            target = find_safe_frontier(
                self.local_map.explored, 
                self.local_map.zone_mask,
                self.local_map.grid, 
                self.current_pos, 
                self.risk_scorer.calculate
            )
            if target:
                new_path = astar_with_hazard(self.local_map.grid, hazard_cost_map, self.current_pos, target)
                self.path = new_path[1:] if new_path else []
                await self._send_to_unity("waypoint_list", self.path)

        elif action_idx == 2: #HOLD
            self.hold_streak += 1
            if self.hold_streak >= 3:
                self.hold_streak = 0

                hazard_cost_map = self.local_map.build_hazard_cost_map(hazard_pos, feature_vector)
                target = find_safe_frontier(
                self.local_map.explored, self.local_map.zone_mask, 
                self.local_map.grid, self.current_pos, self.risk_scorer.calculate
                )  

                if target:
                    new_path = astar_with_hazard(self.local_map.grid, hazard_cost_map, self.current_pos, target)
                    self.path = new_path[1:] if new_path else []
                    await self._send_to_unity("waypoint_list", self.path)
            else:
                await self._send_to_unity("hold", {})

        elif action_idx == 3: #MARK_DANGER
            self.hold_streak = 0
            await self.handler.send_to_coordinator(
                "threat_broadcast",
                self.agent_id,
                #payload
                {
                    "x":hazard_pos[0],
                    "z":hazard_pos[1], 
                    "risk_scalar":risk_scalar, 
                    "radius_m":feature_vector["radius"]
                }
            )

        elif action_idx == 4: #REQUEST_REASSIGNMENT
            self.hold_streak = 0
            await self.handler.send_to_coordinator(
                "request_reassignment",
                self.agent_id, 
                {"orphaned_cells": self.assigned_zone}
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
                await self._send_to_unity("waypoint_list", self.path)

    async def _send_to_unity(self, msg_type, payload):
        await self.handler.comms_clent.send_to_unity(
            msg_type=msg_type,
            agent_id=self.agent_id, 
            payload=payload
        )

