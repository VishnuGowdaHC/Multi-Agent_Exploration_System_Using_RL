import time

class HeartbeatMonitor:
    def __init__(self):
        # Tracks the last time a heartbeat was received per agent
        self.agent_heartbeats = {}
        self.log_file = "coordinator_liveness_log.txt"

    def process_heartbeat(self, agent_id, timestamp):
        self.agent_heartbeats[agent_id] = timestamp

    def audit_liveness(self):
        current_time = time.time()
        for agent_id, last_seen in self.agent_heartbeats.items():
            if current_time - last_seen > 3.0:
                self._log_event(f"[{current_time}] Agent {agent_id} timeout - potential mesh mode entry.")

    def _log_event(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")