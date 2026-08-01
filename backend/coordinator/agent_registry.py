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

    def get_nearest_active_agent(self, orphaned_centroid):
        nearest_agent_id = None
        min_distance = float('inf')

        for agent_id, data in self.agents.items():
            if data["status"] == "alive" and data["position"] is not None:
                # Calculates Euclidean distance to find the closest surviving peer
                dist = math.dist(data["position"], orphaned_centroid)
                if dist < min_distance:
                    min_distance = dist
                    nearest_agent_id = agent_id

        return nearest_agent_id

    def get_all_active_agents(self):
        return [aid for aid, data in self.agents.items() if data["status"] == "alive"]