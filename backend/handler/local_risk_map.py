import numpy as np
import math

class LocalRiskMap:
    def __init__(self, physical_width=100, physical_height=100, resolution=0.5):
        self.resolution = resolution
        self.grid_width = int(physical_width / resolution)
        self.grid_height = int(physical_height / resolution)

        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
        self.explored = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        self.zone_mask = np.zeros((self.grid_width, self.grid_height), dtype=bool)

    def world_to_grid(self, x, z):
        grid_x = math.floor(x / self.resolution)
        grid_z = math.floor(z / self.resolution)

        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_z = max(0, min(self.grid_height - 1, grid_z))

        return grid_x, grid_z

    def update_zone_mask(self, zone_cells):
        self.zone_mask.fill(False)

        for x, z in zone_cells:
            if 0 <= x < self.grid_width and 0 <= z < self.grid_height:
                self.zone_mask[x,z] = True

    def mark_explored(self, grid_pos):
        gx, gz = grid_pos
        self.explored[gx, gz] = True

    def mark_impassable(self, grid_pos):
        gx, gz = grid_pos
        self.grid[gx, gz] = 1 # impassable/obstacle

    def get_coverage_pct(self):
        zone_size = np.sum(self.zone_mask)
        if zone_size == 0:
            return 0.0
        explored_in_zone = np.sum(self.zone_mask & self.explored)

        return float(explored_in_zone / zone_size)

    def build_hazard_cost_map(self, hazard_grid_pos, feature_vector):
        cost_map = np.zeros((self.grid_width, self.grid_height), dtype=np.float32)
        hx, hz = hazard_grid_pos

        radius_m = feature_vector["radius"]
        lethality = feature_vector["lethality"]

        radius_cells = int(math.ceil(radius_m / self.resolution))

        for i in range(-radius_cells, radius_cells + 1):
            for j in range(-radius_cells, radius_cells + 1):
                #the area should be in circle x^2 + y^2 = r^2
                if i**2 + j**2 <= radius_cells**2:
                    target_x = hx + i
                    target_z = hz + j

                    if 0 <= target_x < self.grid_width and 0 <= target_z < self.grid_height:
                        cost_map[target_x, target_z] = max(cost_map[target_x, target_z], lethality)

        return cost_map