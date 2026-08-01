import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum

from handler.local_risk_map import LocalRiskMap
from handler.risk_scorer import RiskScorer
from handler.pathfinder import astar, astar_with_hazard, find_frontier, find_safe_frontier
from coordinator.occupancy_grid import OccupancyGrid
from coordinator.voronoi_partition import VoronoiPartitioner

class Cell(IntEnum):
    EMPTY = 0
    OBSTACLE = 1

class Decision(IntEnum):
    CONTINUE = 0
    REROUTE = 1
    HOLD = 2
    MARK_DANGER = 3
    REQUEST_REASSIGNMENT = 4

@dataclass
class Threat:
    x: int
    z: int  
    features: dict

@dataclass
class AgentState:
    x: int
    z: int  
    alive: bool = True
    local_map: LocalRiskMap = None
    path: list = field(default_factory=list)
    awaiting_decision: bool = False
    pending_cell: tuple = None
    threat_history: bool = False
    hold_streak: int = 0

class MockAgentRegistry:
    def __init__(self):
        self.agents = {}
    def get_all_active_agents(self):
        return list(self.agents.keys())

class TrainingSandboxEnv:
    def __init__(self, num_agents=4, grid_size=15, num_obstacles=12, rewards_config=None, seed=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.rng = np.random.default_rng(seed)
        
        rewards = rewards_config or {}
        self.r_explore = rewards.get("r_explore", 10.0)
        self.r_death = rewards.get("r_death", -50.0)
        self.r_overlap = rewards.get("r_overlap", -5.0)
        self.r_unnecessary_retreat = rewards.get("r_unnecessary_retreat", -2.0)
        self.r_risk_exposure = rewards.get("r_risk_exposure", -1.0)
        self.r_hold_penalty = rewards.get("r_hold_penalty", -0.5)

        self.risk_scorer = RiskScorer()
        self.threats = []
        self.agents = []
        
        # Use the actual OccupancyGrid structure
        self.occupancy_grid = OccupancyGrid(width=grid_size, height=grid_size, resolution=1.0)
        self.agent_registry = MockAgentRegistry()

    def reset(self):
        self.occupancy_grid.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._place_obstacles()
        self._place_agents()
        self._place_threats()
        
        # Use the actual Voronoi Partitioner
        partitioner = VoronoiPartitioner(self.occupancy_grid, self.agent_registry)
        zone_assignments = partitioner.compute_partitions()

        for i, agent in enumerate(self.agents):
            agent.local_map = LocalRiskMap(physical_width=self.grid_size, physical_height=self.grid_size, resolution=1.0)
            agent.local_map.grid = np.copy(self.occupancy_grid.grid)
            
            # Map the 1D cell list from Voronoi into a 2D mask
            if i in zone_assignments:
                agent.local_map.update_zone_mask(zone_assignments[i])
            agent.local_map.mark_explored((agent.x, agent.z))

        return self._get_obs()

    def _place_obstacles(self):
        placed = 0
        while placed < self.num_obstacles:
            x, z = self.rng.integers(0, self.grid_size, size=2)
            if self.occupancy_grid.grid[z, x] == Cell.EMPTY:
                self.occupancy_grid.grid[z, x] = Cell.OBSTACLE
                placed += 1

    def _place_agents(self):
        self.agents = []
        self.agent_registry.agents = {}
        for i in range(self.num_agents):
            while True:
                x, z = self.rng.integers(0, self.grid_size, size=2)
                if self.occupancy_grid.grid[z, x] == Cell.EMPTY and not any(a.x == x and a.z == z for a in self.agents):
                    self.agents.append(AgentState(x=x, z=z))
                    # Register mock position for Voronoi logic
                    self.agent_registry.agents[i] = {"position": {"x": x, "z": z}}
                    break

    def _place_threats(self):
        threat_configs = [
            {"lethality": 0.85, "radius": 2.0, "persistence": "static"},
            {"lethality": 0.03, "radius": 2.0, "persistence": "static"},
        ]
        self.threats = []
        for features in threat_configs:
            while True:
                x, z = self.rng.integers(0, self.grid_size, size=2)
                if self.occupancy_grid.grid[z, x] == Cell.OBSTACLE or any(a.x == x and a.z == z for a in self.agents):
                    continue
                self.threats.append(Threat(x=x, z=z, features=features))
                break

    def _get_active_threats_list(self):
        return [{'pos': (t.x, t.z), 'features': t.features} for t in self.threats]

    def _ensure_path(self, i):
        agent = self.agents[i]
        if agent.path:
            return
        
        target = find_frontier(agent.local_map.explored, agent.local_map.zone_mask, agent.local_map.grid, (agent.x, agent.z))
        if target:
            path = astar(agent.local_map.grid, (agent.x, agent.z), target)
            agent.path = path[1:] if path else []

    def tick(self, decision_actions=None):
        decision_actions = decision_actions or {}
        rewards = {i: 0.0 for i in range(self.num_agents)}
        dones = {i: False for i in range(self.num_agents)}
        info = {i: {} for i in range(self.num_agents)}
        decisions_needed = []
        active_threats = self._get_active_threats_list()

        for i, agent in enumerate(self.agents):
            if not agent.alive:
                dones[i] = True
                continue

            if agent.awaiting_decision:
                action = decision_actions.get(i)
                if action is None:
                    decisions_needed.append(i)
                    continue

                target_x, target_z = agent.pending_cell
                agent.awaiting_decision = False
                agent.pending_cell = None
                
                immediate_risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)

                if action != Decision.HOLD:
                    agent.hold_streak = 0

                if action == Decision.CONTINUE:
                    agent.x, agent.z = target_x, target_z
                    if not agent.local_map.explored[agent.z, agent.x]:
                        agent.local_map.mark_explored((agent.x, agent.z))
                        rewards[i] += self.r_explore
                    
                    if immediate_risk > 0 and self.rng.random() < min(immediate_risk, 1.0) * 0.15:
                        agent.alive = False
                        rewards[i] += self.r_death
                        dones[i] = True
                    else:
                        rewards[i] += self.r_risk_exposure

                elif action == Decision.REROUTE:
                    hazard_cost_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
                    for t in active_threats:
                        hazard_cost_map += agent.local_map.build_hazard_cost_map(t['pos'], t['features'])
                        
                    target = find_safe_frontier(
                        agent.local_map.explored, agent.local_map.zone_mask, 
                        agent.local_map.grid, (agent.x, agent.z), 
                        lambda cx, cz: self.risk_scorer.calculate_cell_risk(cx, cz, active_threats)
                    )
                    if target:
                        new_path = astar_with_hazard(agent.local_map.grid, hazard_cost_map, (agent.x, agent.z), target)
                        agent.path = new_path[1:] if new_path else []

                elif action == Decision.HOLD:
                    agent.hold_streak += 1
                    rewards[i] += self.r_unnecessary_retreat if agent.hold_streak >= 3 else self.r_hold_penalty

                info[i]["risk"] = immediate_risk
                continue

            self._ensure_path(i)
            if not agent.path:
                continue

            next_x, next_z = agent.path[0]
            risk_ahead = self.risk_scorer.calculate_cell_risk(next_x, next_z, active_threats)
            
            if risk_ahead > 0.02:
                agent.awaiting_decision = True
                agent.pending_cell = (next_x, next_z)
                agent.threat_history = True
                decisions_needed.append(i)
                continue

            agent.path.pop(0)
            agent.x, agent.z = next_x, next_z
            
            if not agent.local_map.explored[agent.z, agent.x]:
                agent.local_map.mark_explored((agent.x, agent.z))
                rewards[i] += self.r_explore

        # Calculate localized coverage to determine if everyone is done
        local_coverages = [self._get_local_coverage(a) for a in self.agents]
        all_finished = all(cov >= 0.85 or not a.alive for a, cov in zip(self.agents, local_coverages))
        dones["__all__"] = all(dones.values()) or all_finished
        info["decisions_needed"] = decisions_needed

        return self._get_obs(), rewards, dones, info

    def _get_local_coverage(self, agent):
        """Calculates coverage strictly within the agent's assigned Voronoi zone."""
        zone_cells = agent.local_map.zone_mask
        if not np.any(zone_cells):
            return 1.0 # Zone is empty
        explored_in_zone = np.logical_and(agent.local_map.explored, zone_cells)
        return explored_in_zone.sum() / zone_cells.sum()

    def _get_obs(self):
        obs = {}
        active_threats = self._get_active_threats_list()

        for i, agent in enumerate(self.agents):
            if agent.awaiting_decision and agent.pending_cell:
                risk = self.risk_scorer.calculate_cell_risk(*agent.pending_cell, active_threats)
            else:
                risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)
            
            # FIXED: Uses local coverage, making the handler fully independent
            local_coverage_pct = self._get_local_coverage(agent)
                
            obs[i] = np.array([
                agent.x / self.grid_size,
                agent.z / self.grid_size,
                np.tanh(risk),
                local_coverage_pct,
                float(agent.alive),
                float(agent.threat_history),
            ], dtype=np.float32)
            
        return obs