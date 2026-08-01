import numpy as np
import math

class GlobalRiskMap:
    def __init__(self, physical_width, physical_height, resolution):
        self.resolution = resolution
        self.width = int(physical_width / resolution)
        self.height = int(physical_height / resolution)
        self.risk_grid = np.zeros((self.height, self.width), dtype=np.int8)

    def world_to_grid(self, x, z):
        grid_x = math.floor(x / self.resolution)
        grid_z = math.floor(z / self.resolution)

        grid_x = max(0, min(self.width - 1, grid_x))
        grid_z = max(0, min(self.height - 1, grid_z))

        return grid_x, grid_z

    def apply_threat_broadcast(self, hazard_x, hazard_z, risk_scalar, radius_m):
        center_x, center_z = self.world_to_grid(hazard_x, hazard_z)

        radius_in_cells = int(math.ceil(radius_m / self.resolution))

        for i in range(-radius_in_cells, radius_in_cells + 1):
            for j in range(-radius_in_cells, radius_in_cells + 1):
                #the area should be in circle x^2 + y^2 = r^2
                if i**2 + j**2 <= radius_in_cells**2:
                    target_x = center_x + i
                    target_z = center_z + j

                    if 0 <= target_x < self.width and 0 <= target_z < self.height:
                        self.risk_grid[target_z][target_x] = max(self.risk_grid[target_x][target_z], risk_scalar)
