import time
import math

class AgentRegistry:
    def __init__(self):
        self.agents = {}

    def register_agent(self, agent_id, initial_position=None):
        self.agents[agent_id] = {
            "status": "alive",
            "last_seen_timestamp": time.time(),
            "assigned_zone": None, 
            "position": initial_position 
        }

    def update_heartbeat(self, agent_id, timestamp):
        if agent_id in self.agents:
            self.agents[agent_id]["last_seen_timestamp"] = timestamp

    def update_zone(self, agent_id, zone_data):
        if agent_id in self.agents:
            self.agents[agent_id]["assigned_zone"] = zone_data

    def update_position(self, agent_id, position):
        if agent_id in self.agents:
            self.agents[agent_id]["position"] = position

    def mark_agent_dead(self, agent_id):
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = "dead"


    def get_all_active_agents(self):
        return [aid for aid, data in self.agents.items() if data["status"] == "alive"]