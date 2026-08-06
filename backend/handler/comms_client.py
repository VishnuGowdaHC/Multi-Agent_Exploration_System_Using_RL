import json
import asyncio

class CommsClient:
    def __init__(self):
        self.coordinator_ws = None
        self.unity_ws = None

    def set_coordinator_ws(self, ws):
        self.coordinator_ws = ws

    def set_unity_ws(self, ws):
        self.unity_ws = ws

    async def send_to_coordinator(self, msg_type, agent_id, payload):
        if self.coordinator_ws and self.coordinator_ws.close_code is None:
            envelope = {
                "type": msg_type,
                "agent_id": agent_id,
                "payload": payload
            }
            await self.coordinator_ws.send(json.dumps(envelope))

    async def send_to_unity(self, msg_type, agent_id, payload):
        if self.unity_ws and self.unity_ws.close_code is None:
            envelope = {
                "type": msg_type,
                "agent_id": agent_id,
                "payload": payload
            }
            await self.unity_ws.send(json.dumps(envelope))