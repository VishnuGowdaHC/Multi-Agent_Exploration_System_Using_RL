import uvicorn

from agent_registry import AgentRegistry
from occupancy_grid import GlobalOccupancyGrid
from voronoi_partition import VoronoiPartitioner
from risk_map import GlobalRiskMap
from reassignment import ReassignmentHandler
from heartbeat_monitor import HeartbeatMonitor
from metrics_analyzer import MetricsAnalyzer
from ws_server import CoordinatorWSServer

def create_app():
    print("Booting coordinator...")

    agent_registry = AgentRegistry()
    occupancy_grid = GlobalOccupancyGrid(
        physical_width=100,
        physical_height=100,
        resolution=0.5
    )
    
    risk_map = GlobalRiskMap(
        physical_width=100,
        physical_height=100,
        resolution=0.5)
    
    voronoi_partitioner = VoronoiPartitioner(occupancy_grid, agent_registry)

    reassignment_handler = ReassignmentHandler(
        agent_registry=agent_registry,
        occupancy_grid=occupancy_grid,
        voronoi_partitioner=voronoi_partitioner)
    
    heartbeat_monitor = HeartbeatMonitor()

    ws_server = CoordinatorWSServer(
        registry=agent_registry, 
        occ_grid=occupancy_grid, 
        voronoi=voronoi_partitioner, 
        risk_map=risk_map, 
        reassignment=reassignment_handler,
        heartbeat=heartbeat_monitor
    )
    
    metrics_analyzer = MetricsAnalyzer()
    

    return ws_server.app

app = create_app()

if __name__ == "__main__":
    print("Starting Uvicorn ASGI server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8765, log_level="info")

