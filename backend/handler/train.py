import yaml
import numpy as np
from pathlib import Path
import torch
import os
import csv

from .rl_policy import DQNAgent
from .replay_buffer import ReplayBuffer
from .training_sandbox import TrainingSandboxEnv

class TrainingFramework:
    def __init__(self, config_path="../config/rewards.yml", num_agents=4):
        self.config_path = Path(__file__).parent / config_path
        self.num_agents = num_agents
        self.rewards, self.hyperparams = self._load_config()
        
        self.env = TrainingSandboxEnv(
            num_agents=self.num_agents,
            grid_size=15, 
            num_obstacles=12,
            rewards_config=self.rewards
        )
        
        self.agent = DQNAgent(state_dim=6, action_dim=4, config=self.hyperparams)
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
        ticks_since_decision = {i: 0 for i in range(n)} 
        
        losses = []
        avg_qs = [] # Added for Q-value tracking
        deaths = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0} 
        stuck_overrides = 0

        for t in range(400):
            obs, step_rewards, dones, info = self.env.tick(pending_actions)
            pending_actions = {}
            
            stuck_overrides = info.get("stuck_overrides", 0)

            for i in range(n):
                accumulated_reward[i] += step_rewards[i] 
                ticks_since_decision[i] += 1
                
                done_flag = dones[i] or dones.get("__all__", False)
                
                if train and last_action[i] is not None and done_flag:
                    norm_reward = accumulated_reward[i] / max(1, ticks_since_decision[i])
                    self.replay_buffer.push(last_state[i], last_action[i], norm_reward, obs[i], float(done_flag))
                    last_action[i] = None

            for i in info.get("decisions_needed", []):
                if train and last_action[i] is not None:
                    norm_reward = accumulated_reward[i] / max(1, ticks_since_decision[i])
                    self.replay_buffer.push(last_state[i], last_action[i], norm_reward, obs[i], 0.0)

                accumulated_reward[i] = 0.0
                ticks_since_decision[i] = 0
                
                state = obs[i]
                action = self.agent.act(state, epsilon if train else 0.0)
                
                action_counts[action] += 1
                
                pending_actions[i] = action
                last_state[i] = state
                last_action[i] = action

            if train and len(self.replay_buffer) >= self.hyperparams.get("batch_size", 512):
                # Unpack both loss and avg_q from train_step
                loss, avg_q = self.agent.train_step(self.replay_buffer, self.hyperparams.get("batch_size", 512))
                if loss is not None:
                    losses.append(loss)
                    avg_qs.append(avg_q) # Store the Q-value

            if dones.get("__all__", False):
                break

        coverages = [self.env._get_local_coverage(a) for a in self.env.agents]
        avg_coverage = sum(coverages) / len(coverages)
        global_coverage = self.env.get_global_coverage()
        deaths = sum(1 for a in self.env.agents if not a.alive)  # ground truth, computed BEFORE use
        survival_rate = max(0.0, (n - deaths) / n)

        return {
            "avg_coverage": avg_coverage,
            "global_coverage": global_coverage,
            "avg_loss": float(np.mean(losses)) if losses else None,
            "avg_q": float(np.mean(avg_qs)) if avg_qs else None, # Return the average Q-value
            "survival_rate": survival_rate,
            "action_counts": action_counts,
            "stuck_overrides": stuck_overrides
        }

    def train(self, num_episodes=3000):
        consecutive_successes = 0
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        csv_path = "logs/training_metrics.csv"
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Add Avg_Q to CSV headers
            writer.writerow(["Episode", "Epsilon", "Avg_Coverage", "Global_Coverage", "Survival_Rate", "Avg_Loss", "Avg_Q", "Stuck_Overrides"])

        window_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        window_stuck_overrides = 0

        for ep in range(num_episodes):
            epsilon = max(0.05, 1.0 - (ep / (num_episodes * 0.33)))
            result = self.run_episode(epsilon=epsilon, train=True)
            
            for k, v in result["action_counts"].items():
                window_action_counts[k] += v
            window_stuck_overrides += result["stuck_overrides"]
            
            if ep % 5 == 0:
                self.agent.update_target()

            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    ep, 
                    round(epsilon, 3), 
                    round(result['avg_coverage'], 4), 
                    round(result['global_coverage'], 4),
                    round(result['survival_rate'], 4), 
                    round(result['avg_loss'] or 0.0, 4),
                    round(result['avg_q'] or 0.0, 4), # Write Avg_Q to CSV
                    result['stuck_overrides']
                ])

            if ep % 50 == 0:
                # Print Avg_Q to console
                print(f"[Ep {ep:4d}] Eps: {epsilon:.2f} | Avg Local Coverage: {result['avg_coverage']:.2%} | Global Coverage: {result['global_coverage']:.2%} | Avg survival rate: {result['survival_rate']:.2%} | Avg Loss: {result['avg_loss'] or 0:.4f} | Avg Q: {result['avg_q'] or 0:.2f}")
                
                action_names = {0: "CONTINUE", 1: "REROUTE", 2: "MARK_DANGER", 3: "REASSIGN"}
                total_actions = sum(window_action_counts.values())
                
                if total_actions > 0:
                    dist_str = "  ".join([f"{action_names[k]}: {v/total_actions*100:.1f}%" for k, v in window_action_counts.items()])
                else:
                    dist_str = "No decisions made"
                    
                print(f"          Action dist (last 50 ep, n={total_actions}): {dist_str} | Stuck Overrides: {window_stuck_overrides}")
                
                window_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                window_stuck_overrides = 0

                self.agent.save(self.checkpoint_dir / f"framework_v1_ep_{ep}.pt")

            if result["global_coverage"] > 0.85 and result["survival_rate"] > 0.90:
                consecutive_successes += 1
                if consecutive_successes >= 100:    
                    print(f"Target coverage and survival reached on episode {ep}. Saving model...")
                    self.agent.save(self.checkpoint_dir / f"framework_v1_final_{ep}.pt")
                    break
            else:
                consecutive_successes = 0

        self.agent.save(self.final_path)

if __name__ == "__main__":
    framework = TrainingFramework(num_agents=4)
    framework.train()