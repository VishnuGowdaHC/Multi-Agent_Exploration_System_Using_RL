import heapq
from collections import deque
import numpy as np

# 4-way movement (up, down, left, right)
NEIGHBORS_4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def manhattan(a, b):
    """Heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _reconstruct_path(came_from, current):
    """Backtracks from the goal to the start to build the waypoint list."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(grid, start, goal):
    """
    Standard obstacle-only A*.
    Used for normal frontier navigation and obstacle_blocked replans.
    This is pure replanning, no RL involvement.
    """
    h, w = grid.shape[0], grid.shape[1]
    open_set = [(manhattan(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return _reconstruct_path(came_from, current)

        cx, cy = current
        for dx, dy in NEIGHBORS_4:
            nx, ny = cx + dx, cy + dy
            
            # Boundary and obstacle checks
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[nx, ny] != 0:  # 1 == Obstacle
                continue
                
            neighbor = (nx, ny)
            tentative_g = g_score[current] + 1
            
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None

def astar_with_hazard(grid, hazard_cost_map, start, goal, hazard_weight=25.0):
    """
    A* with an additional per-cell hazard cost layered on top of obstacles[cite: 7].
    Used ONLY when the DQN policy outputs a "Reroute" decision.
    """
    h, w = grid.shape[0], grid.shape[1]
    open_set = [(manhattan(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return _reconstruct_path(came_from, current)

        cx, cy = current
        for dx, dy in NEIGHBORS_4:
            nx, ny = cx + dx, cy + dy
            
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[nx, ny] != 0:
                continue
                
            neighbor = (nx, ny)
            
            # Apply the hazard penalty to the movement cost[cite: 7]
            step_cost = 1.0 + (hazard_weight * hazard_cost_map[nx, ny])
            tentative_g = g_score[current] + step_cost
            
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None

def find_frontier(explored, zone_mask, grid, current_pos):
    """
    BFS outward from current_pos to find the nearest unexplored, passable
    cell that belongs to this agent's Voronoi zone.
    """
    h, w = grid.shape[0], grid.shape[1]
    visited = {current_pos}
    q = deque([current_pos])

    while q:
        cx, cy = q.popleft()
        
        # If unexplored, within zone, and not the standing cell
        if not explored[cx, cy] and zone_mask[cx, cy] and (cx, cy) != current_pos:
            return (cx, cy)

        for dx, dy in NEIGHBORS_4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[nx, ny] != 0:
                continue
            if (nx, ny) in visited:
                continue
                
            visited.add((nx, ny))
            q.append((nx, ny))

    return None

def find_safe_frontier(explored, zone_mask, grid, current_pos, risk_fn, risk_threshold=0.02):
    """
    BFS that skips candidate cells exceeding the risk_threshold.
    Ensures a "Reroute" decision targets a genuinely different, lower-risk 
    destination rather than the same hazardous cell.
    """
    h, w = grid.shape[0], grid.shape[1]
    visited = {current_pos}
    q = deque([current_pos])
    fallback = None

    while q:
        cx, cy = q.popleft()
        
        if not explored[cx, cy] and zone_mask[cx, cy] and (cx, cy) != current_pos:
            if fallback is None:
                fallback = (cx, cy)
                
            # Only return the target if it is mathematically safe
            if risk_fn(cx, cy) <= risk_threshold:
                return (cx, cy)

        for dx, dy in NEIGHBORS_4:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[nx, ny] != 0:
                continue
            if (nx, ny) in visited:
                continue
                
            visited.add((nx, ny))
            q.append((nx, ny))

    return fallback