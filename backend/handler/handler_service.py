import asyncio
import json
import websockets
from .agent_task import AgentTask
#from mesh_manager import MeshManager
from .comms_client import CommsClient
from .global_visualizer import AsyncGlobalMap

class AgentHandlerService:
    def __init__(self, num_agents, coordinator_uri="ws://127.0.0.1:8000/ws", unity_port=8766):
        self.num_agents = num_agents
        self.coordinator_uri = coordinator_uri
        self.unity_port = unity_port
        

        self.agent_tasks = {}
        self.comms_client = CommsClient()
        self.global_map = AsyncGlobalMap(grid_size=30)
        self.agent_start_positions = {}
        #self.mesh_manager = MeshManager()

    async def start(self):
        print(f"Starting Agent Handler Service for {self.num_agents} agents...")

        for i in range(self.num_agents):
            self.agent_tasks[i] = AgentTask(agent_id=i, handler_service=self)

        #Coordinator
        asyncio.create_task(self._connect_to_coordinator())

        #Unity
        print("Loading Unity...")
        async with websockets.serve(self._handle_unity_connection, "127.0.0.1", self.unity_port, subprotocols=None):
            await asyncio.Future()

    async def _connect_to_coordinator(self):
        try:
            async with websockets.connect(self.coordinator_uri) as ws:
                self.comms_client.set_coordinator_ws(ws)
                print("Connected to Coordinator WebSocket")

                if hasattr(self, 'global_map'):
                    asyncio.create_task(self.global_map.start())

                asyncio.create_task(self._heartbeat_loop())
                await self._listen_to_coordinator(ws)

        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
            print(f"Error connecting to Coordinator WebSocket")
            #self.mesh_manager.evaluate_connection_loss()

    async def _handle_unity_connection(self, ws):
        print("Unity WebSocket connection accepted, setting up...")
        self.comms_client.set_unity_ws(ws)
        print("Connected to Unity WebSocket")

        try:
            async for message in ws:
                if message == "ping":
                    continue  # heartbeat, not a JSON envelope

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[handler] Ignoring non-JSON message: {message!r}")
                    continue

                await self._route_unity_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("Unity WebSocket closed")
            self.comms_client.set_unity_ws(None)

    async def _listen_to_coordinator(self, ws):
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type")
            agent_id = data.get("agent_id")
            payload = data.get("payload", {})

            if msg_type == "heartbeat_ack":
                ...
                #self.mesh_manager.recieve_heartbeat_ack()

            if msg_type == "zone_assignment":
                if agent_id in self.agent_tasks:
                    await self.agent_tasks[agent_id].inbox.put({
                        "type": "zone_update",
                        "cells": payload.get("zone_cells")
                    })
                    
            if msg_type == "occupancy_update":
                await self.comms_client.send_to_unity("occupancy_update", agent_id, payload)

    async def _route_unity_message(self, data):
        if data.get("type") == "heartbeat":
            return 
            
        agent_id = data.get("agent_id")
        
        # --- NEW CODE: Trap initial positions from Unity ---
        if data.get("type") == "waypoint_reached" and agent_id not in self.agent_start_positions:
            pos_data = data["payload"]["pos"]
            # Handle both list [x, z] and dict {"x": x, "z": z}
            raw_x = pos_data['x'] if isinstance(pos_data, dict) else pos_data[0]
            raw_z = pos_data['z'] if isinstance(pos_data, dict) else pos_data[1]
            
            self.agent_start_positions[agent_id] = {"x": raw_x, "z": raw_z}
            print(f"Captured start position for Agent {agent_id}: {self.agent_start_positions[agent_id]}")
            
            # Once all agents report in, fire the payload to the Coordinator
            if len(self.agent_start_positions) == self.num_agents:
                await self._send_handler_ready()
        # ---------------------------------------------------

        if agent_id in self.agent_tasks:
            await self.agent_tasks[agent_id].inbox.put(data)

    # --- NEW METHOD: Build the payload dynamically ---
    async def _send_handler_ready(self):
        payload = {
            "agents": [
                {
                    "id": i, 
                    "start_coords": self.agent_start_positions[i]
                } for i in range(self.num_agents)
            ]
        }
        await self.comms_client.send_to_coordinator("handler_ready", "system", payload)
        print("All real agent coordinates verified. Sent handler_ready to Coordinator.")
    async def _heartbeat_loop(self):
        while True:
            await self.comms_client.send_to_coordinator(
                "heartbeat",
                "system",
                {"timestamp": asyncio.get_event_loop().time()}
            )
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    handler = AgentHandlerService(num_agents=4)
    asyncio.run(handler.start())
        