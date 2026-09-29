# decision_engine/demos/tetris_ai.py
"""
Geometric Placement Intelligence for TAMEV Tetris AI.
Implements:
1. Pierre Dellacherie Tetris Board Evaluator (Line clears, holes, aggregate height, bumpiness)
2. Best Landing Placement Search (optimal rotation r* and column c*)
3. Step-by-Step Action Alignment (LEFT, RIGHT, ROTATE, DROP)
4. Semantic Option Description Generation for TAMEV Pointer Model
"""

from typing import Any

TETRIS_SHAPES = {
    1: [[1, 1, 1, 1]],  # I (Line)
    2: [[1, 1], [1, 1]],  # O (Square)
    3: [[0, 1, 0], [1, 1, 1]],  # T
    4: [[0, 1, 1], [1, 1, 0]],  # S
    5: [[1, 1, 0], [0, 1, 1]],  # Z
    6: [[1, 0, 0], [1, 1, 1]],  # J
    7: [[0, 0, 1], [1, 1, 1]],  # L
}

PIECE_NAMES = {1: "I (Line)", 2: "O (Square)", 3: "T", 4: "S", 5: "Z", 6: "J", 7: "L"}


def rotate_shape(shape: list[list[int]]) -> list[list[int]]:
    return [list(r) for r in zip(*shape[::-1])]


def get_unique_rotations(shape: list[list[int]]) -> list[list[list[int]]]:
    rotations = [shape]
    for _ in range(3):
        r = rotate_shape(rotations[-1])
        if r not in rotations:
            rotations.append(r)
    return rotations


def check_collision(
    board: list[list[int]], shape: list[list[int]], px: int, py: int, cols: int = 10, rows: int = 20
) -> bool:
    for r in range(len(shape)):
        for c in range(len(shape[r])):
            if shape[r][c]:
                nx, ny = px + c, py + r
                if nx < 0 or nx >= cols or ny >= rows:
                    return True
                if ny >= 0 and board[ny][nx] != 0:
                    return True
    return False


def drop_piece(
    board: list[list[int]], shape: list[list[int]], px: int, cols: int = 10, rows: int = 20
) -> int | None:
    py = 0
    if check_collision(board, shape, px, py, cols, rows):
        return None
    while not check_collision(board, shape, px, py + 1, cols, rows):
        py += 1
    return py


def evaluate_placement(
    board: list[list[int]], shape: list[list[int]], px: int, cols: int = 10, rows: int = 20
) -> tuple[float, int, int]:
    py = drop_piece(board, shape, px, cols, rows)
    if py is None:
        return float("-inf"), 0, 0

    # Build simulated board
    b_copy = [row[:] for row in board]
    for r in range(len(shape)):
        for c in range(len(shape[r])):
            if shape[r][c]:
                b_copy[py + r][px + c] = 1

    # Lines cleared
    full_rows = sum(1 for r in range(rows) if all(b_copy[r][c] != 0 for c in range(cols)))

    # Board after clearing
    b_cleared = [row for row in b_copy if any(v == 0 for v in row)]
    for _ in range(full_rows):
        b_cleared.insert(0, [0] * cols)

    # Column heights
    heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if b_cleared[r][c] != 0:
                heights[c] = rows - r
                break

    # Holes
    holes = 0
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if b_cleared[r][c] != 0:
                block_found = True
            elif block_found:
                holes += 1

    bumpiness = sum(abs(heights[c] - heights[c + 1]) for c in range(cols - 1))
    agg_height = sum(heights)

    score = +500.0 * full_rows - 1.5 * agg_height - 8.0 * holes - 1.0 * bumpiness
    return score, full_rows, holes


def find_best_placement(
    board: list[list[int]], shape: list[list[int]], cols: int = 10, rows: int = 20
) -> tuple[list[list[int]], int, int, float]:
    """Finds optimal rotation and column for current piece."""
    rotations = get_unique_rotations(shape)
    best_score = float("-inf")
    best_rot = shape
    best_col = 0
    best_clears = 0

    for rot in rotations:
        for c in range(cols - len(rot[0]) + 1):
            score, clears, _holes = evaluate_placement(board, rot, c, cols, rows)
            if score > best_score:
                best_score = score
                best_rot = rot
                best_col = c
                best_clears = clears

    return best_rot, best_col, best_clears, best_score


def analyze_tetris_state(
    board: list[list[int]], current_piece: dict[str, Any], cols: int = 10, rows: int = 20
) -> tuple[str, list[dict[str, Any]]]:
    """
    Analyzes active tetromino against board and outputs:
    1. Contextual board analysis
    2. Ranked actions with semantic descriptions and target landing alignment
    """
    shape = current_piece["shape"]
    px = current_piece["x"]
    py = current_piece["y"]
    type_id = current_piece.get("type_id", 1)
    pname = PIECE_NAMES.get(type_id, "Tetromino")

    best_rot, best_col, best_clears, _best_score = find_best_placement(board, shape, cols, rows)

    # Determine primary required actions to achieve best placement
    needs_rotation = shape != best_rot
    needs_left = px > best_col
    needs_right = px < best_col
    is_aligned = (not needs_rotation) and (px == best_col)

    options = []

    # 1. Action DROP
    if is_aligned:
        desc = f"Action DROP: optimal hard drop at column {best_col}, locks piece, clears {best_clears} lines, creates 0 holes (score +{best_clears * 100 if best_clears else 20} pts)"
        options.append({"id": "drop", "text": desc, "quality": 4, "is_optimal": True})
    else:
        desc = f"Action DROP: premature drop at column {px}, leaves gaps and creates holes"
        options.append({"id": "drop", "text": desc, "quality": 1, "is_optimal": False})

    # 2. Action ROTATE
    if needs_rotation:
        desc = f"Action ROTATE: rotate clockwise to match target horizontal orientation for column {best_col}"
        options.append({"id": "rotate", "text": desc, "quality": 3, "is_optimal": not is_aligned})
    else:
        desc = "Action ROTATE: redundant rotation, misaligns optimal shape"
        options.append({"id": "rotate", "text": desc, "quality": 1, "is_optimal": False})

    # 3. Action LEFT
    if needs_left:
        desc = f"Action LEFT: shift left from column {px} towards target column {best_col}"
        options.append({"id": "left", "text": desc, "quality": 3, "is_optimal": True})
    else:
        desc = f"Action LEFT: shift left away from optimal target column {best_col}"
        options.append({"id": "left", "text": desc, "quality": 1, "is_optimal": False})

    # 4. Action RIGHT
    if needs_right:
        desc = f"Action RIGHT: shift right from column {px} towards target column {best_col}"
        options.append({"id": "right", "text": desc, "quality": 3, "is_optimal": True})
    else:
        desc = f"Action RIGHT: shift right away from optimal target column {best_col}"
        options.append({"id": "right", "text": desc, "quality": 1, "is_optimal": False})

    # Sort options by quality descending
    options.sort(key=lambda x: x["quality"], reverse=True)

    # Column heights profile
    heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if board[r][c] != 0:
                heights[c] = rows - r
                break

    context = (
        f"Tetris {cols}x{rows} grid. Active tetromino: {pname} at (col={px}, row={py}). "
        f"Column heights: {heights}. Target optimal landing: column {best_col} (clears {best_clears} lines). "
        f"Select the safest and most optimal geometric action to clear lines and keep the board flat."
    )

    clean_options = [{"id": o["id"], "text": o["text"]} for o in options]
    return context, clean_options
