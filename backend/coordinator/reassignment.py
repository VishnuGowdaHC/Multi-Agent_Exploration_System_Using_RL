import math

class ReassignmentHandler:
    def __init__(self, agent_registry, occupancy_grid, voronoi_partitioner):
        self.agent_registry = agent_registry
        self.occupancy_grid = occupancy_grid
        self.voronoi_partitioner = voronoi_partitioner

    def handle_reassignment(self, agent_id, orphaned_cells, is_fatal=True):
        if is_fatal:
            self.agent_registry.mark_agent_dead(agent_id)

        active_agents = self.agent_registry.get_all_active_agents()

        if not active_agents:
            return None, {}

        if not orphaned_cells:
            return None, self.voronoi_partitioner.compute_partitions()

        x_coords = [cell[0] for cell in orphaned_cells]
        y_coords = [cell[1] for cell in orphaned_cells]
        centroid = (sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords))

        nearest_agent_id = self._get_nearest_active_agent(centroid, active_agents)

        new_zone_assignment = self.voronoi_partitioner.compute_partitions()

        return nearest_agent_id, new_zone_assignment

    def _get_nearest_active_agent(self, orphaned_centroid, active_agents):
        nearest_agent_id = None
        min_distance = float('inf')

        for agent_id in active_agents:
            pos = self.agent_registry.agents[agent_id]["position"]

            if pos:
                grid_x, grid_z = self.occupancy_grid.world_to_grid(pos['x'], pos['z'])
                dist = math.dist((grid_x, grid_z), orphaned_centroid)
                if dist < min_distance:
                    min_distance = dist
                    nearest_agent_id = agent_id

        return nearest_agent_id
    