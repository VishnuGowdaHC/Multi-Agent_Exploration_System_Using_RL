import numpy as np
import math

class GlobalOccupancyGrid:
    def __init__(self, physical_width, physical_height, resolution):
        self.resolution = resolution
        self.width = int(physical_width / resolution)
        self.height = int(physical_height / resolution)
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def world_to_grid(self, x, z):
        grid_x = math.floor(x / self.resolution)
        grid_z = math.floor(z / self.resolution)

        grid_x = max(0, min(self.width - 1, grid_x))
        grid_z = max(0, min(self.height - 1, grid_z))

        return grid_x, grid_z

    def update_cells(self, state_deltas):
        
        for delta in state_deltas:
            g_x, g_z = self.world_to_grid(delta["x"], delta["z"])

            # 0 = Unexplored, 1 = Explored, -1 = Impassable/Obstacle
            self.grid[g_z][g_x] = delta["state"]

    def get_unexplored_cells(self):
        unexplored_indices = np.argwhere(self.grid == 0)
        return [(int(x), int(z)) for z, x in unexplored_indices]
