import json
import matplotlib.pyplot as plt
import pandas as pd

class MetricsAnalyzer:
    def __init__(self, log_filepath="simulation_data.json"):
        self.log_filepath = log_filepath
        self.event_log = []

    def log_event(self, event_type, timestamp, payload):
        """
        Passively subscribes to the event stream: agent_failure, threat_broadcast, 
        zone_assignment, mesh_mode_entered/exited.
        """
        event = {
            "type": event_type,
            "timestamp": timestamp,
            "data": payload
        }
        self.event_log.append(event)
        self._save_to_disk()

    def _save_to_disk(self):
        """Dumps the event log to JSON for offline plotting."""
        with open(self.log_filepath, 'w') as f:
            json.dump(self.event_log, f, indent=4)

    def plot_training_vs_heuristics(self, training_json, heuristics_json):
        """
        Plots survival rate and exploration coverage % comparing the RL training 
        against the baseline heuristics.
        """
        # Load the external JSON data
        with open(training_json, 'r') as f:
            train_data = json.load(f)
        with open(heuristics_json, 'r') as f:
            heuristic_data = json.load(f)

        # Convert to Pandas DataFrames for easy rolling mean calculation
        df_train = pd.DataFrame(train_data)
        df_heur = pd.DataFrame(heuristic_data)

        # Apply a 50-episode rolling mean as specified in the evaluation
        df_train['coverage_50_mean'] = df_train['coverage_percent'].rolling(window=50).mean()
        df_train['survival_50_mean'] = df_train['survival_rate'].rolling(window=50).mean()
        
        df_heur['coverage_50_mean'] = df_heur['coverage_percent'].rolling(window=50).mean()
        df_heur['survival_50_mean'] = df_heur['survival_rate'].rolling(window=50).mean()

        # Generate the plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Coverage Percentage
        ax1.plot(df_train['episode'], df_train['coverage_50_mean'], label='RL Training Coverage', color='blue')
        ax1.plot(df_heur['episode'], df_heur['coverage_50_mean'], label='Heuristics Coverage', color='cyan', linestyle='--')
        ax1.axhline(y=75.0, color='r', linestyle=':', label='Target Convergence (75%)')
        ax1.set_title('Exploration Coverage % (50-ep Rolling Mean)')
        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Coverage Percent')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Survival Rate
        ax2.plot(df_train['episode'], df_train['survival_50_mean'], label='RL Training Survival', color='darkred')
        ax2.plot(df_heur['episode'], df_heur['survival_50_mean'], label='Heuristics Survival', color='salmon', linestyle='--')
        ax2.axhline(y=95.0, color='g', linestyle=':', label='Target Convergence (95%)')
        ax2.set_title('Survival Rate % (50-ep Rolling Mean)')
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel('Survival Percent')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()