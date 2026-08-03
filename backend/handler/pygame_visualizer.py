import pygame
import yaml
import colorsys
import re
from pathlib import Path

from .training_sandbox import TrainingSandboxEnv, Cell
from .rl_policy import DQNAgent

CELL_SIZE = 40
PANEL_WIDTH = 320
RENDER_FPS = 60  # keep high for smooth input/quit handling, NOT the sim speed

# --- Dark theme palette ---
BG = (18, 18, 22)
PANEL_BG = (24, 24, 30)
GRID_LINE = (45, 45, 52)
UNEXPLORED_FILL = (30, 30, 36)
OBSTACLE_FILL = (70, 40, 40)
TEXT_PRIMARY = (230, 230, 235)
TEXT_DIM = (140, 140, 150)
TEXT_ACCENT = (120, 200, 255)
DEAD_MARK = (200, 60, 60)
THREAT_HIGH = (255, 70, 70)
THREAT_LOW = (255, 210, 90)
PAUSED_TEXT = (255, 200, 80)


def generate_agent_colors(num_agents):
    """Zone colors are dim/desaturated (background); agent colors are bright (foreground)."""
    agent_colors, zone_colors_unexplored, zone_colors_explored = [], [], []
    for i in range(num_agents):
        hue = i / max(num_agents, 1)

        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        agent_colors.append((int(r * 255), int(g * 255), int(b * 255)))

        # Unexplored zone tint: barely visible, just enough to show ownership
        zr, zg, zb = colorsys.hsv_to_rgb(hue, 0.35, 0.16)
        zone_colors_unexplored.append((int(zr * 255), int(zg * 255), int(zb * 255)))

        # Explored zone tint: clearly brighter than unexplored, same hue
        er, eg, eb = colorsys.hsv_to_rgb(hue, 0.45, 0.30)
        zone_colors_explored.append((int(er * 255), int(eg * 255), int(eb * 255)))

    return agent_colors, zone_colors_unexplored, zone_colors_explored


class PygameVisualizer:
    def __init__(self, num_agents=4, ticks_per_second=2.0):
        pygame.init()
        pygame.font.init()

        config_path = Path(__file__).parent / "../config/rewards.yml"
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"

        self.checkpoint_loaded = False
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.hyperparams = config.get("hyperparameters", {})
            self.rewards = config.get("rewards", {})

        self.num_agents = num_agents
        self.agent_colors, self.zone_unexplored, self.zone_explored = generate_agent_colors(self.num_agents)

        self.env = TrainingSandboxEnv(num_agents=self.num_agents, grid_size=15, rewards_config=self.rewards)
        self.obs = self.env.reset()

        self.grid_size = self.env.grid_size
        grid_pixel_size = self.grid_size * CELL_SIZE
        self.screen = pygame.display.set_mode((grid_pixel_size + PANEL_WIDTH, grid_pixel_size))
        pygame.display.set_caption(f"Multi-Agent ({self.num_agents}) Evaluator")
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_med = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 20, bold=True)

        self.agent = DQNAgent(state_dim=6, action_dim=4, config=self.hyperparams)

        # Discover every "framework_v1_ep_{N}.pt" checkpoint on disk, sorted by episode
        # number, so you can step through training progress instead of only ever
        # seeing the final policy.
        self.checkpoints = self._discover_checkpoints()
        self.checkpoint_index = len(self.checkpoints) - 1  # start at the most recent
        self.current_checkpoint_label = "NONE (random policy)"

        if self.checkpoints:
            self._load_checkpoint_at_index(self.checkpoint_index)
        else:
            # Fall back to the final checkpoint if no per-episode ones exist
            final_path = self.checkpoint_dir / "framework_v1_final.pt"
            if final_path.exists():
                self.agent.load(final_path)
                self.checkpoint_loaded = True
                self.current_checkpoint_label = "final"

        # --- Pacing controls (decoupled from render FPS) ---
        self.ticks_per_second = ticks_per_second
        self._tick_accumulator_ms = 0
        self.paused = False
        self.step_once = False

        self.episode_count = 0
        self.last_tick_rewards = {i: 0.0 for i in range(self.num_agents)}

    # ------------------------------------------------------------------
    # Checkpoint discovery / cycling
    # ------------------------------------------------------------------
    def _discover_checkpoints(self):
        """Finds every framework_v1_ep_{N}.pt in checkpoints/, sorted by episode
        number ascending. Also allows the final_{ep} variant train.py saves on
        early convergence, since that's still a real per-episode snapshot."""
        if not self.checkpoint_dir.exists():
            return []

        found = []
        pattern = re.compile(r"framework_v1_(?:ep_|final_)(\d+)\.pt$")
        for p in self.checkpoint_dir.glob("framework_v1_*.pt"):
            match = pattern.search(p.name)
            if match:
                found.append((int(match.group(1)), p))

        found.sort(key=lambda pair: pair[0])
        return found

    def _load_checkpoint_at_index(self, index):
        if not self.checkpoints:
            return
        index = max(0, min(len(self.checkpoints) - 1, index))
        self.checkpoint_index = index
        ep_num, path = self.checkpoints[index]

        # Fresh agent instance so old optimizer/network state can't bleed
        # between checkpoints — load() only restores weights, not identity.
        self.agent = DQNAgent(state_dim=6, action_dim=4, config=self.hyperparams)
        self.agent.load(path)
        self.checkpoint_loaded = True
        self.current_checkpoint_label = f"ep {ep_num}  ({index + 1}/{len(self.checkpoints)})"

    def _cycle_checkpoint(self, direction):
        if not self.checkpoints:
            return
        self._load_checkpoint_at_index(self.checkpoint_index + direction)
        # Reset the episode so behavior differences aren't muddied by
        # mid-episode state carried over from the previous checkpoint.
        self.episode_count = 0
        self.obs = self.env.reset()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def handle_events(self):
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT and self.paused:
                    self.step_once = True
                elif event.key == pygame.K_UP:
                    self.ticks_per_second = min(20.0, self.ticks_per_second + 0.5)
                elif event.key == pygame.K_DOWN:
                    self.ticks_per_second = max(0.5, self.ticks_per_second - 0.5)
                elif event.key == pygame.K_LEFTBRACKET:
                    self._cycle_checkpoint(-1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._cycle_checkpoint(+1)
                elif event.key == pygame.K_0:
                    self._cycle_checkpoint(len(self.checkpoints))  # clamps to last
        return running

    # ------------------------------------------------------------------
    # Drawing: grid
    # ------------------------------------------------------------------
    def draw_grid(self):
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * CELL_SIZE, z * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                owning_agent = None
                for i, agent in enumerate(self.env.agents):
                    if agent.local_map.zone_mask[z, x]:
                        owning_agent = i
                        break

                is_explored = any(
                    agent.local_map.explored[z, x] for agent in self.env.agents
                )

                if self.env.occupancy_grid.grid[z, x] == Cell.OBSTACLE:
                    pygame.draw.rect(self.screen, OBSTACLE_FILL, rect)
                elif owning_agent is not None:
                    fill = self.zone_explored[owning_agent] if is_explored else self.zone_unexplored[owning_agent]
                    pygame.draw.rect(self.screen, fill, rect)
                else:
                    pygame.draw.rect(self.screen, UNEXPLORED_FILL, rect)

                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

    def draw_threats(self):
        for threat in self.env.threats:
            cx = int(threat.x * CELL_SIZE + CELL_SIZE / 2)
            cz = int(threat.z * CELL_SIZE + CELL_SIZE / 2)
            color = THREAT_HIGH if threat.features["lethality"] > 0.5 else THREAT_LOW
            radius = int((threat.features["radius"] / 1.0) * CELL_SIZE)

            surface = pygame.Surface((self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(surface, (*color, 45), (cx, cz), radius)
            self.screen.blit(surface, (0, 0))
            pygame.draw.circle(self.screen, color, (cx, cz), CELL_SIZE // 4)
            pygame.draw.circle(self.screen, BG, (cx, cz), CELL_SIZE // 4, 2)

    def draw_agents(self):
        for i, agent in enumerate(self.env.agents):
            color = self.agent_colors[i]

            if not agent.alive:
                cx = agent.x * CELL_SIZE + CELL_SIZE // 2
                cz = agent.z * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.line(self.screen, DEAD_MARK, (cx - 10, cz - 10), (cx + 10, cz + 10), 3)
                pygame.draw.line(self.screen, DEAD_MARK, (cx - 10, cz + 10), (cx + 10, cz - 10), 3)
                continue

            if len(agent.path) > 0:
                path_points = [(agent.x * CELL_SIZE + CELL_SIZE // 2, agent.z * CELL_SIZE + CELL_SIZE // 2)]
                for px, pz in agent.path:
                    path_points.append((px * CELL_SIZE + CELL_SIZE // 2, pz * CELL_SIZE + CELL_SIZE // 2))
                if len(path_points) > 1:
                    pygame.draw.lines(self.screen, color, False, path_points, 2)

            cx = agent.x * CELL_SIZE + CELL_SIZE // 2
            cz = agent.z * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, color, (cx, cz), CELL_SIZE // 3)
            pygame.draw.circle(self.screen, BG, (cx, cz), CELL_SIZE // 3, 2)

            if agent.awaiting_decision:
                pygame.draw.circle(self.screen, PAUSED_TEXT, (cx, cz), CELL_SIZE // 6)
                if agent.pending_cell:
                    px, pz = agent.pending_cell
                    pygame.draw.rect(
                        self.screen, PAUSED_TEXT,
                        pygame.Rect(px * CELL_SIZE, pz * CELL_SIZE, CELL_SIZE, CELL_SIZE), 2
                    )

    # ------------------------------------------------------------------
    # Drawing: info panel
    # ------------------------------------------------------------------
    def draw_panel(self):
        grid_pixel_size = self.grid_size * CELL_SIZE
        panel_rect = pygame.Rect(grid_pixel_size, 0, PANEL_WIDTH, grid_pixel_size)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)

        x0 = grid_pixel_size + 16
        y = 14

        def line(text, font=self.font_small, color=TEXT_PRIMARY, dy=20):
            nonlocal y
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += dy

        line(f"EPISODE {self.episode_count}", self.font_large, TEXT_ACCENT, dy=30)

        status = "PAUSED" if self.paused else "RUNNING"
        status_color = PAUSED_TEXT if self.paused else (100, 220, 130)
        line(f"[{status}]  step {self.ticks_per_second:.1f}/s", self.font_med, status_color, dy=26)

        ckpt_color = TEXT_DIM if self.checkpoint_loaded else PAUSED_TEXT
        line(f"checkpoint: {self.current_checkpoint_label}", self.font_small, ckpt_color, dy=24)

        line("HYPERPARAMS", self.font_med, TEXT_ACCENT, dy=22)
        line(f"gamma: {self.hyperparams.get('gamma', '-')}", dy=18)
        line(f"lr: {self.hyperparams.get('learning_rate', '-')}", dy=18)
        line(f"batch: {self.hyperparams.get('batch_size', '-')}", dy=18)
        line(f"target_update: {self.hyperparams.get('target_update_every', '-')}", dy=24)

        line("REWARDS", self.font_med, TEXT_ACCENT, dy=22)
        line(f"explore: +{self.rewards.get('r_explore', '-')}", dy=18)
        line(f"death: {self.rewards.get('r_death', '-')}", dy=18)
        line(f"risk_exposure: {self.rewards.get('r_risk_exposure', '-')}", dy=18)
        line(f"reroute: {self.rewards.get('r_reroute_penalty', '-')}", dy=18)
        line(f"mark_danger: +{self.rewards.get('r_mark_danger', '-')}", dy=18)
        line(f"reassign: {self.rewards.get('r_reassignment', '-')}", dy=24)

        line("AGENTS", self.font_med, TEXT_ACCENT, dy=22)
        for i, agent in enumerate(self.env.agents):
            coverage = self.env._get_local_coverage(agent)
            state = "alive" if agent.alive else "DEAD"
            state_color = TEXT_PRIMARY if agent.alive else DEAD_MARK
            swatch_rect = pygame.Rect(x0, y + 3, 10, 10)
            pygame.draw.rect(self.screen, self.agent_colors[i], swatch_rect)
            surf = self.font_small.render(
                f"  #{i} {state}  cov {coverage:.0%}  stuck {agent.stuck_ticks}", True, state_color
            )
            self.screen.blit(surf, (x0, y))
            y += 20

        y += 8
        line(f"stuck overrides (ep): {self.env.stuck_overrides_this_episode}", self.font_small, TEXT_DIM, dy=18)

        y += 8
        line("SPACE pause  |  ->  step", self.font_small, TEXT_DIM, dy=18)
        line("UP/DOWN  speed", self.font_small, TEXT_DIM, dy=18)
        line("[ / ]  prev/next checkpoint  |  0  latest", self.font_small, TEXT_DIM, dy=18)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        running = True
        pending_actions = {}

        while running:
            running = self.handle_events()
            dt_ms = self.clock.tick(RENDER_FPS)

            should_tick = False
            if self.paused:
                if self.step_once:
                    should_tick = True
                    self.step_once = False
            else:
                self._tick_accumulator_ms += dt_ms
                tick_interval_ms = 1000.0 / self.ticks_per_second
                if self._tick_accumulator_ms >= tick_interval_ms:
                    self._tick_accumulator_ms -= tick_interval_ms
                    should_tick = True

            if should_tick:
                for i, agent in enumerate(self.env.agents):
                    if agent.awaiting_decision and agent.alive:
                        pending_actions[i] = self.agent.act(self.obs[i], epsilon=0.0)

                self.obs, self.last_tick_rewards, dones, _ = self.env.tick(pending_actions)
                pending_actions = {}

                if dones["__all__"]:
                    self.episode_count += 1
                    self.obs = self.env.reset()

            self.screen.fill(BG)
            self.draw_grid()
            self.draw_threats()
            self.draw_agents()
            self.draw_panel()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    visualizer = PygameVisualizer(num_agents=4, ticks_per_second=2.0)
    visualizer.run()