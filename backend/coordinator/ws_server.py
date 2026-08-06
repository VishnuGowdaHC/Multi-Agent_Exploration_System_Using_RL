import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

class CoordinatorWSServer:
    def __init__(self, registry, occ_grid, risk_map, voronoi, reassignment, heartbeat):
        self.app = FastAPI(title="Coordinator Websocket Server")
        self.registry = registry
        self.occ_grid = occ_grid
        self.risk_map = risk_map
        self.voronoi = voronoi
        self.reassignment = reassignment
        self.heartbeat = heartbeat

        self.handler_socket = None
        self._setup_routes()

    def _setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            
            await websocket.accept()

            self.handler_socket = websocket
            print("Agent-Handler connected to Coordinator")

            try:
                while True:
                    msg = await websocket.receive_text()
                    data = json.loads(msg)
                    await self.dispatch_message(data)

            except WebSocketDisconnect:
                print("Agent-Handler disconnected from Coordinator")
                self.handler_socket = None

    async def dispatch_message(self, data):
        msg_type = data.get("type")
        agent_id = data.get("agent_id")
        payload = data.get("payload", {})

        if msg_type == "handler_ready":
            await self._handle_startup(payload)

        elif msg_type == "heartbeat":
            self.heartbeat.process_heartbeat(agent_id, data.get("timestamp"))
            await self.send_message("hearbeat_ack", agent_id, {})

        elif msg_type == "threat_broadcast":
            self.risk_map.apply_threat_broadcast(
                hazard_x = payload['x'],
                hazard_z = payload['z'],
                risk_scalar = payload['risk_scalar'],
                radius_m = payload['radius_m']
            )

        elif msg_type == "agent_failure":
            await self._execute_reassignment(agent_id, payload.get("orphaned_cells", []), is_fatal=True)

        elif msg_type == "request_reassignment":
            await self._execute_reassignment(agent_id, payload.get("orphaned_cells", []), is_fatal=False)

        elif msg_type == "state_sync":
            deltas = payload.get("map_deltas", [])
            self.occ_grid.update_cells(deltas)
            await self.send_message("occupancy_update", agent_id, {"cells": deltas})

    async def _handle_startup(self, payload):
        agents = payload.get("agents", {})
        for agent in agents:
            self.registry.register_agent(agent["id"], agent["start_coords"])

        initial_zones = self.voronoi.compute_partitions()

        for aid, zone_cells in initial_zones.items():
            self.registry.update_zone(aid, zone_cells)
            await self.send_message("zone_assignment", aid, {"zone_cells": zone_cells})

    async def _execute_reassignment(self, agent_id, orphaned_cells, is_fatal):
        nearest_id, new_zones = self.reassignment.handle_reassignment(agent_id, orphaned_cells, is_fatal)

        if nearest_id and nearest_id in new_zones:
            appended_cells = new_zones[nearest_id]
            self.registry.update_zone(nearest_id, appended_cells)
            await self.send_message("zone_assignment", nearest_id, {"zone_cells": appended_cells})

    async def send_message(self, msg_type, agent_id, payload):
        if self.handler_socket:
            envelope = {
                "type": msg_type,
                "agent_id": agent_id,
                "timestamp": asyncio.get_event_loop().time(),
                "payload": payload
            }
            await self.handler_socket.send_text(json.dumps(envelope))