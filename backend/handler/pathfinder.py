import heapq
from collections import deque
import numpy as np

# 4-way movement (dx, dz)
NEIGHBORS_4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(grid, start, goal):
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

        cx, cz = current
        for dx, dz in NEIGHBORS_4:
            nx, nz = cx + dx, cz + dz
            
            if not (0 <= nx < w and 0 <= nz < h):
                continue
            # FIX: Access grid as [z, x]
            if grid[nz, nx] != 0:  
                continue
                
            neighbor = (nx, nz)
            tentative_g = g_score[current] + 1
            
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None

def astar_with_hazard(grid, hazard_cost_map, start, goal, hazard_weight=25.0):
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

        cx, cz = current
        for dx, dz in NEIGHBORS_4:
            nx, nz = cx + dx, cz + dz
            
            if not (0 <= nx < w and 0 <= nz < h):
                continue
            # FIX: Access grid as [z, x]
            if grid[nz, nx] != 0:
                continue
                
            neighbor = (nx, nz)
            # FIX: Access hazard_cost_map as [z, x]
            step_cost = 1.0 + (hazard_weight * hazard_cost_map[nz, nx])
            tentative_g = g_score[current] + step_cost
            
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return None

def find_frontier(explored, zone_mask, grid, current_pos):
    h, w = grid.shape[0], grid.shape[1]
    visited = {current_pos}
    q = deque([current_pos])

    while q:
        cx, cz = q.popleft()
        
        # FIX: Access maps as [z, x]
        if not explored[cz, cx] and zone_mask[cz, cx] and (cx, cz) != current_pos:
            return (cx, cz)

        for dx, dz in NEIGHBORS_4:
            nx, nz = cx + dx, cz + dz
            if not (0 <= nx < w and 0 <= nz < h):
                continue
            # FIX: Access grid as [z, x]
            if grid[nz, nx] != 0:  
                continue
            if (nx, nz) in visited:
                continue
                
            visited.add((nx, nz))
            q.append((nx, nz))

    return None

def find_safe_frontier(explored, zone_mask, grid, current_pos, risk_fn, risk_threshold=0.02):
    h, w = grid.shape[0], grid.shape[1]
    visited = {current_pos}
    q = deque([current_pos])

    while q:
        cx, cz = q.popleft()
        
        # FIX: Access maps as [z, x]
        if not explored[cz, cx] and zone_mask[cz, cx] and (cx, cz) != current_pos:
            return (cx, cz)

        for dx, dz in NEIGHBORS_4:
            nx, nz = cx + dx, cz + dz
            
            if not (0 <= nx < w and 0 <= nz < h):
                continue
            # FIX: Access grid as [z, x]
            if grid[nz, nx] != 0:
                continue
            if (nx, nz) in visited:
                continue
                
            if risk_fn(nx, nz) > risk_threshold:
                continue
                
            visited.add((nx, nz))
            q.append((nx, nz))

    return None