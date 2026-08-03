import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum

from .local_risk_map import LocalRiskMap
from .risk_scorer import RiskScorer
from .pathfinder import astar, astar_with_hazard, find_frontier, find_safe_frontier
from ..coordinator.occupancy_grid import GlobalOccupancyGrid
from ..coordinator.voronoi_partition import VoronoiPartitioner

class Cell(IntEnum):
    EMPTY = 0
    OBSTACLE = 1

class Decision(IntEnum):
    CONTINUE = 0
    REROUTE = 1
    MARK_DANGER = 2
    REQUEST_REASSIGNMENT = 3

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
    marked_threats: set = field(default_factory=set) 
    stuck_ticks: int = 0
    last_position: tuple = None

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
        self.r_death = rewards.get("r_death", -40.0)
        self.r_risk_exposure = rewards.get("r_risk_exposure", -2.0)
        self.r_reroute_penalty = rewards.get("r_reroute_penalty", -1.0)
        self.r_mark_danger = rewards.get("r_mark_danger", 2.0)
        self.r_reassignment = rewards.get("r_reassignment", -10.0)
        self.r_overlap = rewards.get("r_overlap", -1.5)

        self.risk_scorer = RiskScorer()
        self.threats = []
        self.agents = []
        self.stuck_overrides_this_episode = 0  
        
        self.occupancy_grid = GlobalOccupancyGrid(physical_width=grid_size, physical_height=grid_size, resolution=1.0)
        self.agent_registry = MockAgentRegistry()

        # Team-level ground truth: separate from each agent's private local_map.explored.
        # Used to detect overlap (a cell another agent already covered) and to report
        # genuine collective coverage, since per-agent coverage can double-count.
        self.global_explored = np.zeros((grid_size, grid_size), dtype=bool)

    def reset(self):
        self.stuck_overrides_this_episode = 0
        self.occupancy_grid.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.global_explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self._place_obstacles()
        self._place_agents()
        self._place_threats()
        
        partitioner = VoronoiPartitioner(self.occupancy_grid, self.agent_registry)
        zone_assignments = partitioner.compute_partitions()

        for i, agent in enumerate(self.agents):
            agent.local_map = LocalRiskMap(grid_size=self.grid_size, resolution=1.0)
            agent.local_map.grid = np.copy(self.occupancy_grid.grid)
            
            if i in zone_assignments:
                agent.local_map.update_zone_mask(zone_assignments[i])
            agent.local_map.mark_explored((agent.x, agent.z))
            self._mark_globally_explored(agent.x, agent.z)

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

            # Continuous hazard exposure: standing inside (or ending up inside) a
            # lethal radius is dangerous every tick you remain there — not just
            # the tick you stepped in. Without this, an agent that goes idle
            # (no path, or gated out of decisions) sits in a lethal cell forever
            # with zero risk ever assessed.
            standing_risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)
            if standing_risk > 0 and self.rng.random() < min(standing_risk, 1.0) * 0.15:
                agent.alive = False
                rewards[i] += self.r_death
                dones[i] = True
                self._recompute_voronoi_zones()
                continue

            if agent.last_position == (agent.x, agent.z):
                agent.stuck_ticks += 1
            else:
                agent.stuck_ticks = 0
                agent.last_position = (agent.x, agent.z)

            if agent.awaiting_decision:
                action = decision_actions.get(i)
                if action is None:
                    decisions_needed.append(i)
                    continue

                if agent.stuck_ticks >= 6:
                    action = Decision.CONTINUE
                    agent.stuck_ticks = 0
                    self.stuck_overrides_this_episode += 1

                target_x, target_z = agent.pending_cell
                agent.awaiting_decision = False
                agent.pending_cell = None
                
                immediate_risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)

                if action == Decision.CONTINUE:
                    agent.x, agent.z = target_x, target_z
                    
                    # THE FIX: Close the immortality gate so they can be re-evaluated near hazards
                    agent.threat_history = False 
                    
                    if agent.path and agent.path[0] == (target_x, target_z):
                        agent.path.pop(0)

                    if not agent.local_map.explored[agent.z, agent.x]:
                        agent.local_map.mark_explored((agent.x, agent.z))
                        if self.global_explored[agent.z, agent.x]:
                            # Another agent already covered this cell — no net gain for the team
                            rewards[i] += self.r_overlap
                        else:
                            rewards[i] += self.r_explore
                            self._mark_globally_explored(agent.x, agent.z)

                    new_risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)
                    
                    if new_risk > 0 and self.rng.random() < min(new_risk, 1.0) * 0.15:
                        agent.alive = False
                        rewards[i] += self.r_death
                        dones[i] = True
                        self._recompute_voronoi_zones()
                    else:
                        rewards[i] += self.r_risk_exposure

                elif action == Decision.REROUTE:
                    agent.threat_history = False
                    new_route = self._compute_reroute(agent, active_threats)
                    if not new_route:
                        rewards[i] += -15.0 
                    else:
                        rewards[i] += self.r_reroute_penalty 
                    agent.path = new_route

                elif action == Decision.MARK_DANGER:
                    agent.threat_history = False
                    nearby_threat = min(active_threats, key=lambda t: (t['pos'][0] - target_x)**2 + (t['pos'][1] - target_z)**2, default=None)
                    threat_id = nearby_threat['pos'] if nearby_threat else (target_x, target_z)
                    
                    if threat_id not in agent.marked_threats:
                        agent.marked_threats.add(threat_id)
                        rewards[i] += self.r_mark_danger  
                    else:
                        rewards[i] += self.r_risk_exposure 
                        
                    new_route = self._compute_reroute(agent, active_threats)
                    if not new_route:
                        rewards[i] += -15.0
                    else:
                        rewards[i] += self.r_reroute_penalty 
                        
                    agent.path = new_route

                elif action == Decision.REQUEST_REASSIGNMENT:
                    agent.threat_history = False
                    self._recompute_voronoi_zones()
                    new_route = self._compute_reroute(agent, active_threats)
                    if not new_route:
                        rewards[i] += -15.0
                    else:
                        rewards[i] += self.r_reassignment
                    agent.path = new_route

                info[i]["risk"] = immediate_risk
                continue

            self._ensure_path(i)
            if not agent.path:
                continue

            next_x, next_z = agent.path[0]
            risk_ahead = self.risk_scorer.calculate_cell_risk(next_x, next_z, active_threats)
            current_risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)
            
            if risk_ahead < 0.01:
                agent.threat_history = False

            if risk_ahead > 0.02 and risk_ahead > current_risk and not agent.threat_history:
                agent.awaiting_decision = True
                agent.pending_cell = (next_x, next_z)
                agent.threat_history = True
                decisions_needed.append(i)
                continue

            agent.path.pop(0)
            agent.x, agent.z = next_x, next_z
            
            if not agent.local_map.explored[agent.z, agent.x]:
                agent.local_map.mark_explored((agent.x, agent.z))
                if self.global_explored[agent.z, agent.x]:
                    rewards[i] += self.r_overlap
                else:
                    rewards[i] += self.r_explore
                    self._mark_globally_explored(agent.x, agent.z)

        global_cov = self.get_global_coverage()
        all_finished = (global_cov >= 0.85)
        dones["__all__"] = all(dones.values()) or all_finished
        info["decisions_needed"] = decisions_needed
        info["stuck_overrides"] = self.stuck_overrides_this_episode

        return self._get_obs(), rewards, dones, info

    def _get_local_coverage(self, agent):
        zone_cells = agent.local_map.zone_mask
        if not np.any(zone_cells):
            return 0.0 
        explored_in_zone = np.logical_and(agent.local_map.explored, zone_cells)
        return explored_in_zone.sum() / zone_cells.sum()

    def _mark_globally_explored(self, x, z):
        self.global_explored[z, x] = True

    def get_global_coverage(self):
        """Honest team-level coverage: unique cells explored across ALL agents,
        not the mean of each agent's own (possibly overlapping) percentage.
        This is the number that should go in the paper, not avg_coverage."""
        explorable = (self.occupancy_grid.grid == Cell.EMPTY)
        if not np.any(explorable):
            return 1.0
        return float(np.logical_and(self.global_explored, explorable).sum() / explorable.sum())

    def _get_obs(self):
        obs = {}
        active_threats = self._get_active_threats_list()

        for i, agent in enumerate(self.agents):
            if agent.awaiting_decision and agent.pending_cell:
                risk = self.risk_scorer.calculate_cell_risk(*agent.pending_cell, active_threats)
            else:
                risk = self.risk_scorer.calculate_cell_risk(agent.x, agent.z, active_threats)
            
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

    def _compute_reroute(self, agent, active_threats):
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
            return new_path[1:] if new_path else []
        return []

    def _recompute_voronoi_zones(self):
        alive_ids = [i for i, a in enumerate(self.agents) if a.alive]

        self.agent_registry.agents = {
            i: {"position": {"x": self.agents[i].x, "z": self.agents[i].z}}
            for i in alive_ids
        }

        partitioner = VoronoiPartitioner(self.occupancy_grid, self.agent_registry)
        zone_assignments = partitioner.compute_partitions()

        for i in alive_ids:
            self.agents[i].local_map.update_zone_mask(zone_assignments.get(i, []))
            self.agents[i].local_map.explored = np.copy(self.global_explored)