# decision_engine/demos/snake_ai.py
"""
Spatial Navigation and Path Intelligence for TAMEV Snake AI.
Provides:
1. BFS Shortest Path to Food
2. Flood-Fill Reachable Space Calculation (Trap/Dead-End Detection)
3. Anti-Looping and Cycle Breaking
4. Semantic Option Description Generation for TAMEV Pointer Model
"""

from collections import deque
from typing import Any

DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

OPPOSITES = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}


def compute_bfs_path(
    width: int,
    height: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Finds shortest collision-free path from start to target."""
    if start == target:
        return [start]
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        if curr == target:
            return path
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nxt = (curr[0] + dx, curr[1] + dy)
            if (
                0 <= nxt[0] < width
                and 0 <= nxt[1] < height
                and nxt not in obstacles
                and nxt not in visited
            ):
                visited.add(nxt)
                queue.append([*path, nxt])
    return None


def compute_flood_fill(
    width: int,
    height: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    max_cells: int = 150,
) -> int:
    """Calculates number of reachable contiguous free cells from start."""
    queue = deque([start])
    visited = {start} | obstacles
    count = 0
    while queue and count < max_cells:
        curr = queue.popleft()
        count += 1
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nxt = (curr[0] + dx, curr[1] + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return count


def analyze_snake_state(
    snake: list[tuple[int, int]],
    food: tuple[int, int],
    direction: str,
    width: int = 20,
    height: int = 20,
    recent_positions: deque | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Analyzes current board state and produces:
    1. Rich contextual state string
    2. Candidate options with semantic descriptions, distance deltas, and quality scores
    """
    head = snake[0]
    curr_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
    # Body obstacles (snake tail will move unless food is at head, so exclude last segment)
    obstacles = set(snake[:-1])

    recent_set = set(recent_positions) if recent_positions else set()

    # Find BFS path from current head to food
    bfs_path = compute_bfs_path(width, height, obstacles, head, food)
    bfs_first_step = bfs_path[1] if bfs_path and len(bfs_path) > 1 else None

    candidates = []

    for name, (dx, dy) in DIRECTIONS.items():
        if name == OPPOSITES.get(direction):
            continue

        nx, ny = head[0] + dx, head[1] + dy
        is_wall = not (0 <= nx < width and 0 <= ny < height)
        is_body = (nx, ny) in obstacles

        dist = abs(nx - food[0]) + abs(ny - food[1])
        is_closer = dist < curr_dist
        is_on_bfs = (nx, ny) == bfs_first_step

        if is_wall or is_body:
            continue

        # Flood fill from candidate next head
        free_cells = compute_flood_fill(width, height, obstacles, (nx, ny), max_cells=150)
        is_dead_end = free_cells < max(len(snake), 4)
        is_revisit = (nx, ny) in recent_set

        if is_dead_end:
            desc = f"Move {name}: dangerous trap, dead end with only {free_cells} reachable cells"
            quality = 0
        elif (is_on_bfs or is_closer) and not is_revisit:
            desc = f"Move {name}: optimal move, reduces distance to apple to {dist} steps (spacious open path, {free_cells}+ cells free)"
            quality = 4
        elif is_closer and is_revisit:
            desc = f"Move {name}: safe approach to food, distance {dist} steps ({free_cells}+ cells free)"
            quality = 3
        elif not is_revisit:
            desc = f"Move {name}: safe alternative, maintains clearance ({free_cells}+ cells free, distance {dist} steps)"
            quality = 2
        else:
            desc = f"Move {name}: suboptimal move, moves away from apple (distance {dist} steps)"
            quality = 1

        cand = {
            "id": name,
            "text": desc,
            "dist": dist,
            "free_cells": free_cells,
            "quality": quality,
            "is_dead_end": is_dead_end,
        }
        candidates.append(cand)

    # Sort candidates by quality descending (best first for reference)
    candidates.sort(key=lambda x: x["quality"], reverse=True)

    # If all safe moves are exhausted, provide emergency non-colliding moves
    if not candidates:
        for name, (dx, dy) in DIRECTIONS.items():
            if name != OPPOSITES.get(direction):
                nx, ny = head[0] + dx, head[1] + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                    candidates.append(
                        {
                            "id": name,
                            "text": f"Move {name}: emergency evasion move",
                            "dist": abs(nx - food[0]) + abs(ny - food[1]),
                            "free_cells": 1,
                            "quality": 1,
                            "is_dead_end": True,
                        }
                    )

    if not candidates:
        # Complete corner trap
        candidates.append(
            {
                "id": direction,
                "text": f"Move {direction}: forward step",
                "dist": curr_dist,
                "free_cells": 0,
                "quality": 0,
                "is_dead_end": True,
            }
        )

    dx_rel = food[0] - head[0]
    dy_rel = food[1] - head[1]
    context = (
        f"Snake head at ({head[0]}, {head[1]}). Target apple at ({food[0]}, {food[1]}) "
        f"[relative: dx={dx_rel}, dy={dy_rel}]. Current heading: {direction}. "
        f"Snake length: {len(snake)}. Grid: {width}x{height}. "
        f"Select the safest and most optimal directional move to reach the apple without colliding or looping."
    )

    return context, candidates
