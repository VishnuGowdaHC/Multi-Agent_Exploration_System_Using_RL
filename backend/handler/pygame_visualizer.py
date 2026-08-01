import pygame
import yaml
import colorsys
from pathlib import Path

from handler.training_sandbox import TrainingSandboxEnv, Cell
from handler.rl_policy import DQNAgent

CELL_SIZE = 40
FPS = 10
WHITE, BLACK, GRAY, RED, YELLOW = (255, 255, 255), (0, 0, 0), (150, 150, 150), (255, 50, 50), (255, 255, 0)

def generate_agent_colors(num_agents):
    agent_colors, zone_colors = [], []
    for i in range(num_agents):
        hue = i / max(num_agents, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        agent_colors.append((int(r*255), int(g*255), int(b*255)))
        zr, zg, zb = colorsys.hsv_to_rgb(hue, 0.2, 1.0)
        zone_colors.append((int(zr*255), int(zg*255), int(zb*255)))
    return agent_colors, zone_colors

class PygameVisualizer:
    def __init__(self, num_agents=4):
        pygame.init()
        
        config_path = Path(__file__).parent / "../config/rewards.yaml"
        model_path = Path(__file__).parent / "checkpoints/framework_v1_final.pt"
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.hyperparams = config.get("hyperparameters", {})
            self.rewards = config.get("rewards", {})
            
        self.num_agents = num_agents
        self.agent_colors, self.zone_colors = generate_agent_colors(self.num_agents)
        
        self.env = TrainingSandboxEnv(num_agents=self.num_agents, grid_size=15, rewards_config=self.rewards)
        self.obs = self.env.reset()
        
        self.grid_size = self.env.grid_size
        self.screen = pygame.display.set_mode((self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE))
        pygame.display.set_caption(f"Multi-Agent ({self.num_agents}) Evaluator")
        self.clock = pygame.time.Clock()
        
        self.agent = DQNAgent(state_dim=6, action_dim=5, config=self.hyperparams)
        if model_path.exists():
            self.agent.load(model_path)

    def draw_grid(self):
        # FIXED: Iterates using z for vertical bounds to match the architecture
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * CELL_SIZE, z * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Fetch zone mask locally per agent
                for i, agent in enumerate(self.env.agents):
                    if agent.local_map.zone_mask[z, x]:
                        pygame.draw.rect(self.screen, self.zone_colors[i], rect)
                        break
                
                if self.env.occupancy_grid.grid[z, x] == Cell.OBSTACLE:
                    pygame.draw.rect(self.screen, GRAY, rect)
                    
                pygame.draw.rect(self.screen, BLACK, rect, 1)

    def draw_threats(self):
        for threat in self.env.threats:
            cx = int(threat.x * CELL_SIZE + CELL_SIZE / 2)
            cz = int(threat.z * CELL_SIZE + CELL_SIZE / 2)
            color = RED if threat.features["lethality"] > 0.5 else YELLOW
            radius = int((threat.features["radius"] / 1.0) * CELL_SIZE)
            
            surface = pygame.Surface((self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(surface, (*color, 60), (cx, cz), radius)
            self.screen.blit(surface, (0, 0))
            pygame.draw.circle(self.screen, color, (cx, cz), CELL_SIZE // 4)

    def draw_agents(self):
        for i, agent in enumerate(self.env.agents):
            if not agent.alive:
                continue
                
            color = self.agent_colors[i]
            if len(agent.path) > 0:
                path_points = [(agent.x * CELL_SIZE + CELL_SIZE//2, agent.z * CELL_SIZE + CELL_SIZE//2)]
                for px, pz in agent.path:
                    path_points.append((px * CELL_SIZE + CELL_SIZE//2, pz * CELL_SIZE + CELL_SIZE//2))
                if len(path_points) > 1:
                    pygame.draw.lines(self.screen, color, False, path_points, 3)

            cx = agent.x * CELL_SIZE + CELL_SIZE // 2
            cz = agent.z * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, color, (cx, cz), CELL_SIZE // 3)
            
            if agent.awaiting_decision:
                pygame.draw.circle(self.screen, RED, (cx, cz), CELL_SIZE // 6)
                if agent.pending_cell:
                    px, pz = agent.pending_cell
                    pygame.draw.rect(self.screen, RED, pygame.Rect(px * CELL_SIZE, pz * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

    def run(self):
        running = True
        pending_actions = {}
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            for i, agent in enumerate(self.env.agents):
                if agent.awaiting_decision and agent.alive:
                    pending_actions[i] = self.agent.act(self.obs[i], epsilon=0.0)
                    
            self.obs, _, dones, _ = self.env.tick(pending_actions)
            pending_actions = {}

            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_threats()
            self.draw_agents()
            pygame.display.flip()

            if dones["__all__"]:
                pygame.time.wait(2000)
                self.obs = self.env.reset()

            self.clock.tick(FPS)
        pygame.quit()

if __name__ == "__main__":
    visualizer = PygameVisualizer(num_agents=4)
    visualizer.run()