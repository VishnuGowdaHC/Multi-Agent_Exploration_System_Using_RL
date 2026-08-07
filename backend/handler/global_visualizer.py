import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import asyncio

class AsyncGlobalMap:
    # 1. Discrete cell states (Priority hierarchy)
    UNEXPLORED = 0
    EXPLORED = 1
    OBSTACLE = 2
    NEUTRAL = 3
    WARNING = 4  # Mid-tier threat (Wasp)
    LETHAL = 5   # Max-tier threat (Wolf)

    # 2. Solid colors for the grid
    COLORS = [
        "#f4f4f4",  # 0: Unexplored (Light Grey)
        "#bcbcbc",  # 1: Explored (Light Green)
        "#4a4a4a",  # 2: Obstacle (Dark Grey)
        "#017a19",  # 3: Neutral (Solid Blue)
        "#eab308",  # 4: Warning (Solid Yellow)
        "#e53e3e",  # 5: Lethal (Solid Red)
    ]

    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.master_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Dictionary to hold real-time rover positions {agent_id: (grid_x, grid_z)}
        self.agent_positions = {}

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 8), facecolor="#1e1e1e")
        self.ax.set_facecolor("#1e1e1e")
        self.fig.canvas.manager.set_window_title("Live Multi-Agent Tactical Map")
        self.ax.set_title("Live Global Threat & Exploration Map", color="white", fontsize=14, pad=15)

        self.cmap = mcolors.ListedColormap(self.COLORS)
        self.norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], self.cmap.N)
        
        # Render the static environment grid
        self.img = self.ax.imshow(
            self.master_grid, cmap=self.cmap, norm=self.norm,
            origin="lower", interpolation="nearest"
        )
        
        # Render the floating rover layer (Bright Cyan dots with black borders)
        self.rover_scatter = self.ax.scatter([], [], c="#06b6d4", edgecolors="black", s=80, marker="o", zorder=10)

        # --- Legend ---
        legend_handles = [
            Patch(facecolor=self.COLORS[self.EXPLORED], label="Explored"),
            Patch(facecolor=self.COLORS[self.OBSTACLE], label="Obstacle"),
            Patch(facecolor=self.COLORS[self.NEUTRAL], label="Cow"),
            Patch(facecolor=self.COLORS[self.WARNING], label="Wasp"),
            Patch(facecolor=self.COLORS[self.LETHAL], label="Wolf"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#06c6d4", markeredgecolor='black', markersize=9, label='Rover')
        ]
        legend = self.ax.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
            ncol=3, frameon=False, fontsize=10
        )
        for text in legend.get_texts():
            text.set_color("white")

        # --- Formatting ---
        self.ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        self.ax.grid(which="minor", color="#555555", linestyle="-", linewidth=0.3)
        self.ax.tick_params(which="minor", size=0)
        self.ax.tick_params(colors="white")

        self.fig.tight_layout()
        plt.show(block=False)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._render_loop())
        except RuntimeError:
            pass

    async def _render_loop(self):
        """Background task that redraws the plot."""
        while True:
            # 1. Update the static map
            self.img.set_data(self.master_grid)
            
            # 2. Update the dynamic rovers instantly
            if self.agent_positions:
                offsets = [(x, z) for x, z in self.agent_positions.values()]
                self.rover_scatter.set_offsets(offsets)
                
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            await asyncio.sleep(0.1)

    def update_rover_position(self, agent_id, grid_x, grid_z):
        """Called instantly whenever an agent moves."""
        self.agent_positions[agent_id] = (grid_x, grid_z)

    def merge_local_state(self, explored_mask, obstacle_grid, hazard_pos=None, hazard_radius=0, tag=None):
        newly_explored = explored_mask & (self.master_grid == self.UNEXPLORED)
        self.master_grid[newly_explored] = self.EXPLORED
        
        is_obstacle = (obstacle_grid == 1)
        self.master_grid[is_obstacle & (self.master_grid < self.OBSTACLE)] = self.OBSTACLE

        if hazard_pos is not None:
            hx, hz = hazard_pos
            for i in range(-hazard_radius, hazard_radius + 1):
                for j in range(-hazard_radius, hazard_radius + 1):
                    if i**2 + j**2 <= hazard_radius**2:
                        tx, tz = hx + i, hz + j
                        if 0 <= tx < self.grid_size and 0 <= tz < self.grid_size:
                            if tag == "cow":
                                if self.master_grid[tz, tx] < self.NEUTRAL:
                                    self.master_grid[tz, tx] = self.NEUTRAL
                            elif tag == "wasp":
                                if self.master_grid[tz, tx] < self.WARNING:
                                    self.master_grid[tz, tx] = self.WARNING
                            else:
                                self.master_grid[tz, tx] = self.LETHAL

    async def start(self):
        await self._render_loop()