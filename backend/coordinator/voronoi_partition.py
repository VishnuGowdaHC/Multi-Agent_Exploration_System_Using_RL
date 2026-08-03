import numpy as np
from scipy.spatial import Voronoi
from matplotlib.path import Path

class VoronoiPartitioner:
    def __init__(self, occupancy_grid, agent_registry):
        self.occupancy_grid = occupancy_grid
        self.agent_registry = agent_registry

    def compute_partitions(self):
        active_agents = self.agent_registry.get_all_active_agents()

        if len(active_agents) < 2:
            return self._assign_all_to_single_agent(active_agents)

        agent_seeds = []
        agent_ids = []

        for agent_id in active_agents:
            pos = self.agent_registry.agents[agent_id]["position"]

            if pos:
                grid_x, grid_z = self.occupancy_grid.world_to_grid(pos['x'], pos['z'])
                agent_seeds.append((grid_x, grid_z))
                agent_ids.append(agent_id)

        w, h = self.occupancy_grid.width, self.occupancy_grid.height
        dummy_points = [[-w, -h], [w*2, -h], [w*2, h*2], [-w, h*2]]
        all_points = np.vstack([agent_seeds, dummy_points])

        vor = Voronoi(all_points)

        unexplored_cells = self.occupancy_grid.get_unexplored_cells()
        zone_assignments = {aid: [] for aid in agent_ids}

        if not unexplored_cells:
            return zone_assignments

        for i, agent_id in enumerate(agent_ids):
            region_index = vor.point_region[i]
            region_vertices = vor.regions[region_index]

            polygon_point = [vor.vertices[v] for v in region_vertices if v != -1]

            if len(polygon_point) >= 3:
                poly_path = Path(polygon_point)

                for cell in unexplored_cells:
                    if poly_path.contains_point(cell):
                        zone_assignments[agent_id].append(cell)

        return zone_assignments

    def _assign_all_to_single_agent(self, agent_ids):
        if not agent_ids:
            return {}

        unexplored_cells = self.occupancy_grid.get_unexplored_cells()
        return {agent_ids[0]: unexplored_cells}


