import yaml
import numpy as np
from pathlib import Path

from handler.rl_policy import DQNAgent
from handler.replay_buffer import ReplayBuffer
from handler.training_sandbox import TrainingSandboxEnv

class TrainingFramework:
    def __init__(self, config_path="../config/rewards.yaml", num_agents=4):
        self.config_path = Path(__file__).parent / config_path
        self.num_agents = num_agents
        self.rewards, self.hyperparams = self._load_config()
        
        self.env = TrainingSandboxEnv(
            num_agents=self.num_agents,
            grid_size=15, 
            num_obstacles=12,
            rewards_config=self.rewards
        )
        
        self.agent = DQNAgent(state_dim=6, action_dim=5, config=self.hyperparams)
        self.replay_buffer = ReplayBuffer(capacity=self.hyperparams.get("buffer_size", 50000))
        
        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.final_path = self.checkpoint_dir / "framework_v1_final.pt"

    def _load_config(self):
        print(f"Loading experimental parameters from {self.config_path}...")
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config.get("rewards", {}), config.get("hyperparameters", {})
        except FileNotFoundError:
            print("Config not found. Using defaults.")
            return {}, {}

    def run_episode(self, epsilon, train=True):
        obs = self.env.reset()
        n = len(self.env.agents)
        
        pending_actions = {}
        last_state = {i: None for i in range(n)}
        last_action = {i: None for i in range(n)}
        
        accumulated_reward = {i: 0.0 for i in range(n)} 
        losses = []

        for t in range(400):
            obs, step_rewards, dones, info = self.env.tick(pending_actions)
            pending_actions = {}

            for i in range(n):
                accumulated_reward[i] += step_rewards[i] 
                done_flag = dones[i] or dones["__all__"]
                
                if train and last_action[i] is not None and done_flag:
                    self.replay_buffer.push(last_state[i], last_action[i], accumulated_reward[i], obs[i], float(done_flag))
                    last_action[i] = None

            for i in info["decisions_needed"]:
                if train and last_action[i] is not None:
                    self.replay_buffer.push(last_state[i], last_action[i], accumulated_reward[i], obs[i], 0.0)

                accumulated_reward[i] = 0.0
                state = obs[i]
                action = self.agent.act(state, epsilon if train else 0.0)
                pending_actions[i] = action
                last_state[i] = state
                last_action[i] = action

            if train and len(self.replay_buffer) >= self.hyperparams.get("batch_size", 512):
                loss = self.agent.train_step(self.replay_buffer, self.hyperparams.get("batch_size", 512))
                if loss is not None:
                    losses.append(loss)

            if dones["__all__"]:
                break

        # Calculate average local coverage for reporting
        coverages = [self.env._get_local_coverage(a) for a in self.env.agents]
        avg_coverage = sum(coverages) / len(coverages)

        return {
            "avg_coverage": avg_coverage,
            "avg_loss": float(np.mean(losses)) if losses else None,
        }

    def train(self, num_episodes=3000):
        for ep in range(num_episodes):
            epsilon = max(0.05, 1.0 - (ep / (num_episodes * 0.33)))
            result = self.run_episode(epsilon=epsilon, train=True)
            
            if ep % self.hyperparams.get("target_update_every", 5) == 0:
                self.agent.update_target()

            if ep % 50 == 0:
                print(f"[Ep {ep:4d}] Eps: {epsilon:.2f} | Avg Local Coverage: {result['avg_coverage']:.2%} | Loss: {result['avg_loss'] or 0:.4f}")

        self.agent.save(self.final_path)

if __name__ == "__main__":
    framework = TrainingFramework(num_agents=4)
    framework.train()