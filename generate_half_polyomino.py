#!/usr/bin/env python3
"""Generate half-cell polyomino-style tiling puzzles.

The puzzle uses a small-cell grid where one ordinary cell is a 2x2 block.
Every ordinary cell in every board/piece shape must be one of six legal masks:

    bit order: 0bTL TR BL BR
      TL = top-left, TR = top-right, BL = bottom-left, BR = bottom-right

That makes these masks legal:

    0000 empty
    1111 full
    1100 top half
    0011 bottom half
    1010 left half
    0101 right half

All 1/4-cell, 3/4-cell, diagonal-half, and L-shaped artifacts are rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCALE = 2
PIECE_COUNT = 6
PIECE_AREA_SMALL = 16

BOARD_W = 10  # ordinary cells
BOARD_H = 7   # ordinary cells

ALLOW_ROTATE = True
ALLOW_MIRROR = False

TARGET_MIN_SOLUTIONS = 4
TARGET_MAX_SOLUTIONS = 16
SOLUTION_COUNT_LIMIT = 100

# Bit order is 0bTLTRBLBR:
# 0b1000 top-left, 0b0100 top-right, 0b0010 bottom-left, 0b0001 bottom-right.
MASK_EMPTY = 0b0000
MASK_FULL = 0b1111
MASK_TOP = 0b1100
MASK_BOTTOM = 0b0011
MASK_LEFT = 0b1010
MASK_RIGHT = 0b0101

ALLOWED_MASKS = {
    MASK_EMPTY,   # empty
    MASK_FULL,    # full
    MASK_TOP,     # top half
    MASK_BOTTOM,  # bottom half
    MASK_LEFT,    # left half
    MASK_RIGHT,   # right half
}
HALF_MASKS = {MASK_TOP, MASK_BOTTOM, MASK_LEFT, MASK_RIGHT}

MASK_TO_OFFSETS = {
    MASK_EMPTY: (),
    MASK_FULL: ((0, 0), (1, 0), (0, 1), (1, 1)),
    MASK_TOP: ((0, 0), (1, 0)),
    MASK_BOTTOM: ((0, 1), (1, 1)),
    MASK_LEFT: ((0, 0), (0, 1)),
    MASK_RIGHT: ((1, 0), (1, 1)),
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
COLORS = [
    "#df5b57",
    "#4f8bd6",
    "#55a868",
    "#c77cce",
    "#d79542",
    "#4db6ac",
    "#8d6e63",
    "#9e9ac8",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

GUIDED_SUPER_LAYOUTS: list[list[MacroCell]] = [
    # Eight 2x2 ordinary-cell blocks arranged as a non-rectangular board.
    # This layout is deliberately lumpy but still easy to partition cleanly.
    [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2)],
    [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2)],
]

ROBUST_PAIR_SHAPE_TEXTS = [
    # All of these are 16 small cells, legal half-cell masks only, connected,
    # and pass the paper-strength checks.  They are intentionally chunky:
    # no one-small-cell bridge and no dangling small-cell tips.
    "##../####/####/####/##..",
    "....##/..####/######/####..",
    ".##./.##./####/####/..##/..##",
    "###./###./####/####/..##",
    "##.../##.../#####/#####/..##.",
    "####/####/####/..##/..##",
    "..##/..##/..##/####/####/..##",
    "..##/####/####/####/..##",
]


Cell = tuple[int, int]
MacroCell = tuple[int, int]


@dataclass(frozen=True)
class Placement:
    piece_index: int
    cells: frozenset[Cell]
    mask: int
    origin: Cell
    orientation: int


@dataclass
class Analysis:
    solution_count: int
    rotated_solution_count: int
    fixed_pieces: int
    movable_pieces: int
    average_piece_candidates: float
    half_cell_count_per_piece: list[int]
    total_half_cell_count: int
    horizontal_half_cell_count_per_piece: list[int]
    vertical_half_cell_count_per_piece: list[int]
    horizontal_half_cell_count: int
    vertical_half_cell_count: int
    horizontal_half_cell_contacts: int
    vertical_half_cell_contacts: int
    quarter_artifact_count: int
    fragile_artifact_count: int
    duplicate_piece_count: int
    difficulty_score: float


@dataclass
class Candidate:
    board: set[Cell]
    pieces: list[set[Cell]]
    solutions: list[dict[int, frozenset[Cell]]]
    solution_count: int
    placements_by_piece: list[list[Placement]]
    score: float
    analysis: Analysis
    attempts: int


def mask_for_offsets(offsets: Iterable[Cell]) -> int:
    mask = 0
    for dx, dy in offsets:
        if dx == 0 and dy == 0:
            mask |= 0b1000
        elif dx == 1 and dy == 0:
            mask |= 0b0100
        elif dx == 0 and dy == 1:
            mask |= 0b0010
        elif dx == 1 and dy == 1:
            mask |= 0b0001
        else:
            raise ValueError(f"offset outside 2x2 cell: {(dx, dy)}")
    return mask


def cells_to_masks(cells: set[Cell] | frozenset[Cell]) -> dict[MacroCell, int]:
    masks: dict[MacroCell, int] = defaultdict(int)
    for x, y in cells:
        mx, dx = divmod(x, SCALE)
        my, dy = divmod(y, SCALE)
        masks[(mx, my)] |= mask_for_offsets(((dx, dy),))
    return dict(masks)


def masks_to_cells(masks: dict[MacroCell, int]) -> set[Cell]:
    cells: set[Cell] = set()
    for (mx, my), mask in masks.items():
        for dx, dy in MASK_TO_OFFSETS[mask]:
            cells.add((mx * SCALE + dx, my * SCALE + dy))
    return cells


def cells_from_ascii(text: str) -> set[Cell]:
    rows = text.split("/")
    return {
        (x, y)
        for y, row in enumerate(rows)
        for x, char in enumerate(row)
        if char == "#"
    }


def robust_pair_shape_library() -> list[set[Cell]]:
    shapes = [normalize_cells(cells_from_ascii(text)) for text in ROBUST_PAIR_SHAPE_TEXTS]
    valid_shapes = []
    for shape in shapes:
        if (
            len(shape) == PIECE_AREA_SMALL
            and is_legal_half_cell_shape(shape)
            and is_connected(shape)
            and count_half_cells(shape) >= 2
            and count_fragile_artifacts(shape) == 0
        ):
            valid_shapes.append(shape)
    return valid_shapes


def is_legal_half_cell_shape(cells: set[Cell] | frozenset[Cell]) -> bool:
    """
    Check that a small-cell shape contains no 1/4-cell or 3/4-cell artifacts.

    Each ordinary 2x2 cell is converted to a mask.  Only the six masks in
    ALLOWED_MASKS are accepted: empty, full, top, bottom, left, and right.
    """
    return all(mask in ALLOWED_MASKS for mask in cells_to_masks(cells).values())


def count_half_cells(cells: set[Cell] | frozenset[Cell]) -> int:
    """
    Return how many ordinary cells are occupied by a legal half-cell mask.
    """
    return sum(1 for mask in cells_to_masks(cells).values() if mask in HALF_MASKS)


def is_horizontal_half_mask(mask: int) -> bool:
    return mask in (MASK_TOP, MASK_BOTTOM)


def is_vertical_half_mask(mask: int) -> bool:
    return mask in (MASK_LEFT, MASK_RIGHT)


def count_horizontal_half_cells(cells: set[Cell] | frozenset[Cell]) -> int:
    """Return how many ordinary cells use a top/bottom half mask."""
    return sum(1 for mask in cells_to_masks(cells).values() if is_horizontal_half_mask(mask))


def count_vertical_half_cells(cells: set[Cell] | frozenset[Cell]) -> int:
    """Return how many ordinary cells use a left/right half mask."""
    return sum(1 for mask in cells_to_masks(cells).values() if is_vertical_half_mask(mask))


def board_boundary_metrics(cells: set[Cell] | frozenset[Cell]) -> dict[str, float]:
    """Measure exterior board irregularities on the ordinary-cell grid."""
    masks = cells_to_masks(cells)
    if not masks:
        return {
            "board_area_small": 0,
            "board_area_ordinary_equiv": 0.0,
            "occupied_macro_cells": 0,
            "boundary_irregularities": 0,
            "boundary_half_cell_irregularities": 0,
            "boundary_full_cell_irregularities": 0,
        }

    macro_cells = set(masks)
    min_x = min(x for x, _ in macro_cells)
    max_x = max(x for x, _ in macro_cells)
    min_y = min(y for _, y in macro_cells)
    max_y = max(y for _, y in macro_cells)
    half_irregularities = sum(1 for mask in masks.values() if mask in HALF_MASKS)
    missing_adjacent = {
        (x, y)
        for y in range(min_y, max_y + 1)
        for x in range(min_x, max_x + 1)
        if (x, y) not in macro_cells
        and any(neighbor in macro_cells for neighbor in neighbors4((x, y)))
    }
    full_irregularities = len(missing_adjacent)
    return {
        "board_area_small": len(cells),
        "board_area_ordinary_equiv": len(cells) / 4.0,
        "occupied_macro_cells": len(macro_cells),
        "boundary_irregularities": half_irregularities + full_irregularities,
        "boundary_half_cell_irregularities": half_irregularities,
        "boundary_full_cell_irregularities": full_irregularities,
    }


def count_half_cell_orientations(pieces: list[set[Cell]] | list[frozenset[Cell]]) -> dict[str, object]:
    """Return horizontal/vertical half-cell counts across all pieces."""
    horizontal_per_piece = [count_horizontal_half_cells(piece) for piece in pieces]
    vertical_per_piece = [count_vertical_half_cells(piece) for piece in pieces]
    return {
        "horizontal_half_cell_count": sum(horizontal_per_piece),
        "vertical_half_cell_count": sum(vertical_per_piece),
        "horizontal_half_cell_count_per_piece": horizontal_per_piece,
        "vertical_half_cell_count_per_piece": vertical_per_piece,
    }


def count_half_cell_contacts_by_orientation(
    solution: dict[int, frozenset[Cell]],
    pieces: list[set[Cell]] | list[frozenset[Cell]] | None = None,
) -> dict[str, int]:
    """
    Count half-cell small-edge contacts with other pieces by half-mask direction.

    The optional pieces argument keeps the public API aligned with callers that
    evaluate a solution for a known piece set; the placed solution cells carry
    the masks needed for this count.
    """
    del pieces
    owner: dict[Cell, int] = {}
    for piece_index, cells in solution.items():
        for cell in cells:
            owner[cell] = piece_index

    contacts: dict[str, set[tuple[tuple[int, int], tuple[Cell, Cell]]]] = {
        "horizontal": set(),
        "vertical": set(),
    }
    for piece_index, cells in solution.items():
        for (mx, my), mask in cells_to_masks(cells).items():
            if is_horizontal_half_mask(mask):
                orientation = "horizontal"
            elif is_vertical_half_mask(mask):
                orientation = "vertical"
            else:
                continue
            for dx, dy in MASK_TO_OFFSETS[mask]:
                cell = (mx * SCALE + dx, my * SCALE + dy)
                for neighbor in neighbors4(cell):
                    other_piece = owner.get(neighbor)
                    if other_piece is None or other_piece == piece_index:
                        continue
                    piece_pair = tuple(sorted((piece_index, other_piece)))
                    cell_pair = tuple(sorted((cell, neighbor)))
                    contacts[orientation].add((piece_pair, cell_pair))
    return {
        "horizontal": len(contacts["horizontal"]),
        "vertical": len(contacts["vertical"]),
    }


def best_half_cell_contacts_by_orientation(
    solutions: list[dict[int, frozenset[Cell]]],
    pieces: list[set[Cell]] | list[frozenset[Cell]],
) -> dict[str, int]:
    """Return the solution contact counts with the best overall orientation coverage."""
    if not solutions:
        return {"horizontal": 0, "vertical": 0}
    return max(
        (count_half_cell_contacts_by_orientation(solution, pieces) for solution in solutions),
        key=lambda item: (
            item["horizontal"] + item["vertical"],
            min(item["horizontal"], item["vertical"]),
            item["horizontal"],
            item["vertical"],
        ),
    )


def count_quarter_artifacts(cells: set[Cell] | frozenset[Cell]) -> int:
    """Count ordinary cells whose 2x2 mask is not one of the legal masks."""
    return sum(1 for mask in cells_to_masks(cells).values() if mask not in ALLOWED_MASKS)


def has_quarter_artifact(cells: set[Cell] | frozenset[Cell]) -> bool:
    """
    Return True if a shape contains 1/4-cell, 3/4-cell, diagonal-half, or L masks.
    """
    return count_quarter_artifacts(cells) > 0


def small_cell_degree(cells: set[Cell] | frozenset[Cell], cell: Cell) -> int:
    return sum(1 for neighbor in neighbors4(cell) if neighbor in cells)


def has_dangling_small_cell(cells: set[Cell] | frozenset[Cell]) -> bool:
    """Reject paper-weak tips that hang from a single small-cell edge."""
    return any(small_cell_degree(cells, cell) <= 1 for cell in cells)


def has_articulation_small_cell(cells: set[Cell] | frozenset[Cell]) -> bool:
    """Reject shapes where one small-cell is the only bridge between regions."""
    if len(cells) <= 2:
        return False
    original = set(cells)
    for cell in original:
        remaining = original - {cell}
        if remaining and not is_connected(remaining):
            return True
    return False


def has_repeated_half_strip(cells: set[Cell] | frozenset[Cell]) -> bool:
    """
    Reject consecutive half-cell masks that make a half-cell-wide strip.

    A single half-cell tab/notch is acceptable.  Two or more left/right half masks
    stacked vertically, or two or more top/bottom half masks chained horizontally,
    create a paper-thin band and are rejected.
    """
    masks = cells_to_masks(cells)
    for (mx, my), mask in masks.items():
        if mask in (MASK_LEFT, MASK_RIGHT):
            if masks.get((mx, my - 1)) == mask or masks.get((mx, my + 1)) == mask:
                return True
        if mask in (MASK_TOP, MASK_BOTTOM):
            if masks.get((mx - 1, my)) == mask or masks.get((mx + 1, my)) == mask:
                return True
    return False


def count_fragile_artifacts(cells: set[Cell] | frozenset[Cell]) -> int:
    """
    Count paper-weak shape features.

    This is deliberately conservative: paper pieces should not have narrow
    half-cell strings, dangling small-cell ends, or a one-small-cell bridge.
    """
    count = 0
    if has_dangling_small_cell(cells):
        count += 1
    if has_articulation_small_cell(cells):
        count += 1
    if has_repeated_half_strip(cells):
        count += 1
    return count


def duplicate_piece_count(pieces: list[set[Cell]]) -> int:
    signatures = [canonical_signature(piece) for piece in pieces]
    return len(signatures) - len(set(signatures))


def solution_identity_signature(
    solution: dict[int, frozenset[Cell]],
    pieces: list[set[Cell]],
) -> tuple[tuple[tuple[Cell, ...], tuple[tuple[Cell, ...], ...]], ...]:
    """Canonicalize a solution modulo swaps of identical physical cuts."""
    groups: dict[tuple[Cell, ...], list[tuple[Cell, ...]]] = defaultdict(list)
    for piece_index, cells in solution.items():
        piece_sig = canonical_signature(pieces[piece_index])
        groups[piece_sig].append(tuple(sorted(cells)))
    return tuple(
        (piece_sig, tuple(sorted(placements)))
        for piece_sig, placements in sorted(groups.items())
    )


def count_effective_solutions(
    solutions: list[dict[int, frozenset[Cell]]],
    pieces: list[set[Cell]],
) -> int:
    """Count solutions after ignoring pure swaps of identical pieces."""
    return len({solution_identity_signature(solution, pieces) for solution in solutions})


def normalize_cells(cells: set[Cell] | frozenset[Cell]) -> set[Cell]:
    if not cells:
        return set()
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return {(x - min_x, y - min_y) for x, y in cells}


def bounds(cells: set[Cell] | frozenset[Cell]) -> tuple[int, int, int, int]:
    min_x = min(x for x, _ in cells)
    max_x = max(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_y = max(y for _, y in cells)
    return min_x, min_y, max_x, max_y


def is_connected(cells: set[Cell] | frozenset[Cell]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in cells and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return len(seen) == len(cells)


def is_macro_connected(cells: set[MacroCell] | frozenset[MacroCell]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in cells and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return len(seen) == len(cells)


def has_macro_hole(cells: set[MacroCell]) -> bool:
    min_x = min(x for x, _ in cells) - 1
    max_x = max(x for x, _ in cells) + 1
    min_y = min(y for _, y in cells) - 1
    max_y = max(y for _, y in cells) + 1
    start = (min_x, min_y)
    outside = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                continue
            if (nx, ny) in cells or (nx, ny) in outside:
                continue
            outside.add((nx, ny))
            queue.append((nx, ny))
    for y in range(min_y + 1, max_y):
        for x in range(min_x + 1, max_x):
            if (x, y) not in cells and (x, y) not in outside:
                return True
    return False


def has_small_hole(cells: set[Cell]) -> bool:
    min_x = min(x for x, _ in cells) - 1
    max_x = max(x for x, _ in cells) + 1
    min_y = min(y for _, y in cells) - 1
    max_y = max(y for _, y in cells) + 1
    start = (min_x, min_y)
    outside = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                continue
            if (nx, ny) in cells or (nx, ny) in outside:
                continue
            outside.add((nx, ny))
            queue.append((nx, ny))
    for y in range(min_y + 1, max_y):
        for x in range(min_x + 1, max_x):
            if (x, y) not in cells and (x, y) not in outside:
                return True
    return False


def macro_to_full_small(cells: set[MacroCell] | frozenset[MacroCell]) -> set[Cell]:
    masks = {cell: MASK_FULL for cell in cells}
    return masks_to_cells(masks)


def random_macro_board(
    rng: random.Random,
    width: int,
    height: int,
    area: int,
    allow_holes: bool,
    max_tries: int = 500,
) -> set[MacroCell] | None:
    if area > width * height:
        return None

    center = (width // 2, height // 2)
    all_cells = {(x, y) for y in range(height) for x in range(width)}
    for _ in range(max_tries):
        start = (
            min(width - 1, max(0, center[0] + rng.randint(-2, 2))),
            min(height - 1, max(0, center[1] + rng.randint(-1, 1))),
        )
        cells = {start}
        frontier = {
            (nx, ny)
            for nx, ny in neighbors4(start)
            if 0 <= nx < width and 0 <= ny < height
        }

        while len(cells) < area and frontier:
            # Prefer compact-but-lumpy growth near existing cells.
            candidates = list(frontier)
            weights = []
            for c in candidates:
                neighbor_count = sum(1 for n in neighbors4(c) if n in cells)
                distance = abs(c[0] - center[0]) + abs(c[1] - center[1])
                weights.append(1.0 + neighbor_count * 3.0 + max(0, 8 - distance) * 0.12)
            chosen = rng.choices(candidates, weights=weights, k=1)[0]
            frontier.remove(chosen)
            cells.add(chosen)
            for n in neighbors4(chosen):
                if n not in cells and n in all_cells:
                    frontier.add(n)

        if len(cells) != area:
            continue
        if not is_macro_connected(cells):
            continue
        if not allow_holes and has_macro_hole(cells):
            continue
        min_x = min(x for x, _ in cells)
        max_x = max(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        max_y = max(y for _, y in cells)
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        if bbox_w < 5 or bbox_h < 4:
            continue
        if bbox_w * bbox_h == area:
            continue
        fill_ratio = area / (bbox_w * bbox_h)
        if fill_ratio < 0.48 or fill_ratio > 0.88:
            continue
        return cells
    return None


def neighbors4(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def connected_subsets_of_size(
    board: set[MacroCell],
    size: int = 4,
) -> list[frozenset[MacroCell]]:
    """Enumerate connected macro-cell subsets of a given size."""
    subsets: set[frozenset[MacroCell]] = set()
    ordered = sorted(board)
    for start in ordered:
        stack: list[tuple[frozenset[MacroCell], frozenset[MacroCell]]] = [
            (frozenset({start}), frozenset(n for n in neighbors4(start) if n in board))
        ]
        while stack:
            shape, frontier = stack.pop()
            if len(shape) == size:
                subsets.add(shape)
                continue
            if not frontier:
                continue
            for cell in list(frontier):
                new_shape = set(shape)
                new_shape.add(cell)
                # This ordering guard avoids producing every subset from every
                # possible start while still allowing non-monotone shapes.
                if min(new_shape) != start:
                    continue
                new_frontier = set(frontier)
                new_frontier.remove(cell)
                for n in neighbors4(cell):
                    if n in board and n not in new_shape:
                        new_frontier.add(n)
                stack.append((frozenset(new_shape), frozenset(new_frontier)))
    return sorted(subsets, key=lambda s: (min(s), sorted(s)))


def partition_into_tetrominoes(
    board: set[MacroCell],
    piece_count: int,
    rng: random.Random,
    max_nodes: int = 30_000,
) -> list[set[MacroCell]] | None:
    subsets = connected_subsets_of_size(board, 4)
    by_cell: dict[MacroCell, list[frozenset[MacroCell]]] = defaultdict(list)
    for subset in subsets:
        for cell in subset:
            by_cell[cell].append(subset)
    for items in by_cell.values():
        rng.shuffle(items)

    nodes = 0

    def search(remaining: frozenset[MacroCell], chosen: list[frozenset[MacroCell]]):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return None
        if not remaining:
            return chosen
        if len(chosen) >= piece_count:
            return None

        best_cell = None
        best_options: list[frozenset[MacroCell]] | None = None
        for cell in remaining:
            options = [s for s in by_cell[cell] if s <= remaining]
            if not options:
                return None
            if best_options is None or len(options) < len(best_options):
                best_cell = cell
                best_options = options
                if len(options) == 1:
                    break
        assert best_cell is not None and best_options is not None
        options = best_options[:]
        rng.shuffle(options)
        for subset in options:
            result = search(frozenset(remaining - subset), chosen + [subset])
            if result is not None:
                return result
        return None

    result = search(frozenset(board), [])
    if result is None or len(result) != piece_count:
        return None
    return [set(s) for s in result]


def region_map(regions: list[set[MacroCell]]) -> dict[MacroCell, int]:
    mapping: dict[MacroCell, int] = {}
    for index, region in enumerate(regions):
        for cell in region:
            mapping[cell] = index
    return mapping


def make_swapped_pieces(
    regions: list[set[MacroCell]],
    rng: random.Random,
    min_half_cells: int = 2,
    max_tries: int = 250,
) -> list[set[Cell]] | None:
    mapping = region_map(regions)
    edges: list[tuple[MacroCell, MacroCell, int, int, str]] = []
    for cell, p in mapping.items():
        x, y = cell
        for n, direction in (((x + 1, y), "h"), ((x, y + 1), "v")):
            q = mapping.get(n)
            if q is not None and q != p:
                edges.append((cell, n, p, q, direction))

    if not edges:
        return None

    for _ in range(max_tries):
        rng.shuffle(edges)
        selected: list[tuple[MacroCell, MacroCell, int, int, str, bool]] = []
        used_cells: set[MacroCell] = set()
        covered: set[int] = set()

        # First cover every piece with at least one half-cell swap.
        for c, d, p, q, direction in edges:
            if c in used_cells or d in used_cells:
                continue
            if p in covered and q in covered:
                continue
            selected.append((c, d, p, q, direction, rng.choice((False, True))))
            used_cells.add(c)
            used_cells.add(d)
            covered.add(p)
            covered.add(q)
            if len(covered) == len(regions):
                break

        if len(covered) != len(regions):
            continue

        # Add a few extra legal-looking offsets, still avoiding cell conflicts.
        for c, d, p, q, direction in edges:
            if c in used_cells or d in used_cells:
                continue
            if rng.random() > 0.22:
                continue
            selected.append((c, d, p, q, direction, rng.choice((False, True))))
            used_cells.add(c)
            used_cells.add(d)

        masks_by_piece: list[dict[MacroCell, int]] = []
        for region in regions:
            masks_by_piece.append({cell: MASK_FULL for cell in region})

        for c, d, p, q, direction, flip in selected:
            if direction == "h":
                p_mask, q_mask = (MASK_TOP, MASK_BOTTOM) if not flip else (MASK_BOTTOM, MASK_TOP)
            else:
                p_mask, q_mask = (MASK_LEFT, MASK_RIGHT) if not flip else (MASK_RIGHT, MASK_LEFT)
            masks_by_piece[p][c] = p_mask
            masks_by_piece[p][d] = p_mask
            masks_by_piece[q][c] = q_mask
            masks_by_piece[q][d] = q_mask

        pieces = [masks_to_cells(masks) for masks in masks_by_piece]
        if all(validate_piece(piece, min_half_cells=min_half_cells) for piece in pieces):
            canonical = [canonical_signature(piece) for piece in pieces]
            # Exact duplicates make solution counting less meaningful for a puzzle.
            if len(set(canonical)) < max(2, len(canonical) - 1):
                continue
            return [normalize_cells(piece) for piece in pieces]
    return None


def validate_piece(piece: set[Cell], min_half_cells: int = 2) -> bool:
    if len(piece) != PIECE_AREA_SMALL:
        return False
    if not is_connected(piece):
        return False
    if not is_legal_half_cell_shape(piece):
        return False
    if count_half_cells(piece) < min_half_cells:
        return False
    if count_fragile_artifacts(piece) != 0:
        return False
    min_x, min_y, max_x, max_y = bounds(piece)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if max(width, height) > 12 and min(width, height) <= 2:
        return False
    return True


def transform_cells(
    cells: set[Cell] | frozenset[Cell],
    rotation: int,
    mirror: bool = False,
) -> set[Cell]:
    transformed = set()
    for x, y in cells:
        tx = -x if mirror else x
        ty = y
        r = rotation % 4
        if r == 0:
            nx, ny = tx, ty
        elif r == 1:
            nx, ny = ty, -tx
        elif r == 2:
            nx, ny = -tx, -ty
        else:
            nx, ny = -ty, tx
        transformed.add((nx, ny))
    return normalize_cells(transformed)


def orientations(
    cells: set[Cell] | frozenset[Cell],
    allow_rotate: bool,
    allow_mirror: bool,
) -> list[set[Cell]]:
    rotations = range(4) if allow_rotate else range(1)
    mirrors = (False, True) if allow_mirror else (False,)
    seen: set[tuple[Cell, ...]] = set()
    result: list[set[Cell]] = []
    for mirror in mirrors:
        for rotation in rotations:
            oriented = transform_cells(cells, rotation, mirror)
            if not is_legal_half_cell_shape(oriented):
                continue
            sig = tuple(sorted(oriented))
            if sig in seen:
                continue
            seen.add(sig)
            result.append(oriented)
    return result


def canonical_signature(cells: set[Cell] | frozenset[Cell]) -> tuple[Cell, ...]:
    variants = orientations(cells, allow_rotate=True, allow_mirror=False)
    if not variants:
        return tuple(sorted(normalize_cells(cells)))
    return min(tuple(sorted(v)) for v in variants)


def normalize_board_and_pieces(
    board: set[Cell],
    pieces: list[set[Cell]],
) -> tuple[set[Cell], list[set[Cell]]]:
    min_x, min_y, _, _ = bounds(board)
    board2 = {(x - min_x, y - min_y) for x, y in board}
    return board2, [normalize_cells(piece) for piece in pieces]


def prefer_landscape(
    board: set[Cell],
    pieces: list[set[Cell]],
) -> tuple[set[Cell], list[set[Cell]]]:
    min_x, min_y, max_x, max_y = bounds(board)
    if (max_y - min_y) <= (max_x - min_x):
        return board, pieces
    rotated_board = transform_cells(board, 1, False)
    rotated_pieces = [transform_cells(piece, 1, False) for piece in pieces]
    return normalize_board_and_pieces(rotated_board, rotated_pieces)


def enumerate_placements(
    board: set[Cell],
    pieces: list[set[Cell]],
    allow_rotate: bool,
    allow_mirror: bool,
) -> tuple[list[list[Placement]], dict[Cell, int], int]:
    board_cells = sorted(board)
    index = {cell: i for i, cell in enumerate(board_cells)}
    board_mask = (1 << len(board_cells)) - 1
    _, _, board_max_x, board_max_y = bounds(board)

    placements_by_piece: list[list[Placement]] = []
    for piece_index, piece in enumerate(pieces):
        piece_placements: list[Placement] = []
        for orientation_index, oriented in enumerate(orientations(piece, allow_rotate, allow_mirror)):
            _, _, piece_max_x, piece_max_y = bounds(oriented)
            for ty in range(0, board_max_y - piece_max_y + 1, SCALE):
                for tx in range(0, board_max_x - piece_max_x + 1, SCALE):
                    placed = frozenset((x + tx, y + ty) for x, y in oriented)
                    if not placed <= board:
                        continue
                    mask = 0
                    for cell in placed:
                        mask |= 1 << index[cell]
                    piece_placements.append(
                        Placement(
                            piece_index=piece_index,
                            cells=placed,
                            mask=mask,
                            origin=(tx, ty),
                            orientation=orientation_index,
                        )
                    )
        # Stable de-duplication can matter for symmetric pieces.
        unique: dict[int, Placement] = {}
        for placement in piece_placements:
            unique.setdefault(placement.mask, placement)
        placements_by_piece.append(list(unique.values()))
    return placements_by_piece, index, board_mask


def count_solutions(
    board: set[Cell],
    pieces: list[set[Cell]],
    allow_rotate: bool,
    allow_mirror: bool,
    limit: int,
) -> tuple[int, list[dict[int, frozenset[Cell]]], list[list[Placement]]]:
    placements_by_piece, index, board_mask = enumerate_placements(
        board, pieces, allow_rotate, allow_mirror
    )
    if any(not placements for placements in placements_by_piece):
        return 0, [], placements_by_piece

    cell_to_placements: dict[int, list[tuple[int, Placement]]] = defaultdict(list)
    for piece_index, placements in enumerate(placements_by_piece):
        for placement in placements:
            m = placement.mask
            while m:
                bit = m & -m
                cell_index = bit.bit_length() - 1
                cell_to_placements[cell_index].append((piece_index, placement))
                m ^= bit

    remaining_start = frozenset(range(len(pieces)))
    solutions: list[dict[int, frozenset[Cell]]] = []
    count = 0

    def search(occupied: int, remaining: frozenset[int], chosen: dict[int, Placement]) -> None:
        nonlocal count
        if count >= limit:
            return
        if not remaining:
            if occupied == board_mask:
                count += 1
                if len(solutions) < limit:
                    solutions.append({p: placement.cells for p, placement in chosen.items()})
            return

        empty_mask = board_mask & ~occupied
        best_cell_index = None
        best_options: list[tuple[int, Placement]] | None = None
        m = empty_mask
        while m:
            bit = m & -m
            cell_index = bit.bit_length() - 1
            options = [
                (p, placement)
                for p, placement in cell_to_placements[cell_index]
                if p in remaining and (placement.mask & occupied) == 0
            ]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_cell_index = cell_index
                best_options = options
                if len(options) == 1:
                    break
            m ^= bit

        assert best_cell_index is not None and best_options is not None
        # Deterministic order keeps seeded runs reproducible.
        best_options.sort(key=lambda item: (item[0], item[1].origin, item[1].orientation))
        for piece_index, placement in best_options:
            chosen[piece_index] = placement
            search(
                occupied | placement.mask,
                frozenset(p for p in remaining if p != piece_index),
                chosen,
            )
            chosen.pop(piece_index, None)
            if count >= limit:
                return

    search(0, remaining_start, {})
    return count, solutions, placements_by_piece


def analyze_candidate(
    board: set[Cell],
    pieces: list[set[Cell]],
    solutions: list[dict[int, frozenset[Cell]]],
    solution_count: int,
    rotated_solution_count: int,
    placements_by_piece: list[list[Placement]],
    min_solutions: int,
    max_solutions: int,
) -> Analysis:
    half_counts = [count_half_cells(piece) for piece in pieces]
    total_half = sum(half_counts)
    orientation_counts = count_half_cell_orientations(pieces)
    horizontal_half_count = int(orientation_counts["horizontal_half_cell_count"])
    vertical_half_count = int(orientation_counts["vertical_half_cell_count"])
    horizontal_per_piece = list(orientation_counts["horizontal_half_cell_count_per_piece"])
    vertical_per_piece = list(orientation_counts["vertical_half_cell_count_per_piece"])
    contacts_by_orientation = best_half_cell_contacts_by_orientation(solutions, pieces)
    quarter_count = count_quarter_artifacts(board) + sum(
        count_quarter_artifacts(piece) for piece in pieces
    )
    fragile_count = sum(count_fragile_artifacts(piece) for piece in pieces)
    duplicate_count = duplicate_piece_count(pieces)
    fixed = 0
    if solutions:
        for piece_index in range(len(pieces)):
            seen_positions = {solution[piece_index] for solution in solutions if piece_index in solution}
            if len(seen_positions) == 1:
                fixed += 1
    movable = len(pieces) - fixed
    avg_candidates = sum(len(p) for p in placements_by_piece) / max(1, len(placements_by_piece))
    score = score_candidate(
        board=board,
        pieces=pieces,
        solution_count=solution_count,
        min_solutions=min_solutions,
        max_solutions=max_solutions,
        fixed_pieces=fixed,
        average_piece_candidates=avg_candidates,
        total_half_cell_count=total_half,
        horizontal_half_cell_count=horizontal_half_count,
        vertical_half_cell_count=vertical_half_count,
        horizontal_half_cell_contacts=contacts_by_orientation["horizontal"],
        vertical_half_cell_contacts=contacts_by_orientation["vertical"],
        quarter_artifact_count=quarter_count,
        fragile_artifact_count=fragile_count,
        duplicate_piece_count=duplicate_count,
    )
    return Analysis(
        solution_count=solution_count,
        rotated_solution_count=rotated_solution_count,
        fixed_pieces=fixed,
        movable_pieces=movable,
        average_piece_candidates=avg_candidates,
        half_cell_count_per_piece=half_counts,
        total_half_cell_count=total_half,
        horizontal_half_cell_count_per_piece=horizontal_per_piece,
        vertical_half_cell_count_per_piece=vertical_per_piece,
        horizontal_half_cell_count=horizontal_half_count,
        vertical_half_cell_count=vertical_half_count,
        horizontal_half_cell_contacts=contacts_by_orientation["horizontal"],
        vertical_half_cell_contacts=contacts_by_orientation["vertical"],
        quarter_artifact_count=quarter_count,
        fragile_artifact_count=fragile_count,
        duplicate_piece_count=duplicate_count,
        difficulty_score=score,
    )


def score_candidate(
    board: set[Cell],
    pieces: list[set[Cell]],
    solution_count: int,
    min_solutions: int,
    max_solutions: int,
    fixed_pieces: int,
    average_piece_candidates: float,
    total_half_cell_count: int,
    quarter_artifact_count: int,
    fragile_artifact_count: int,
    duplicate_piece_count: int,
    horizontal_half_cell_count: int = 0,
    vertical_half_cell_count: int = 0,
    horizontal_half_cell_contacts: int = 0,
    vertical_half_cell_contacts: int = 0,
) -> float:
    if quarter_artifact_count:
        return -10_000.0 - quarter_artifact_count * 1000
    if fragile_artifact_count:
        return -8_000.0 - fragile_artifact_count * 1000
    if duplicate_piece_count:
        return -6_000.0 - duplicate_piece_count * 1000
    target_mid = (min_solutions + max_solutions) / 2
    score = 120.0 - abs(solution_count - target_mid) * 5.0
    score += min(total_half_cell_count, len(pieces) * 4) * 4.0
    score += min(horizontal_half_cell_count, vertical_half_cell_count) * 50.0
    score -= abs(horizontal_half_cell_count - vertical_half_cell_count) * 20.0
    if horizontal_half_cell_count == 0:
        score -= 500.0
    if vertical_half_cell_count == 0:
        score -= 500.0
    score += horizontal_half_cell_contacts * 12.0
    score += vertical_half_cell_contacts * 12.0
    score += max(0, len(pieces) - fixed_pieces) * 8.0

    macro_board = {(x // SCALE, y // SCALE) for x, y in board}
    min_x = min(x for x, _ in macro_board)
    max_x = max(x for x, _ in macro_board)
    min_y = min(y for _, y in macro_board)
    max_y = max(y for _, y in macro_board)
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    bbox_area = bbox_w * bbox_h
    fill_ratio = len(macro_board) / bbox_area
    aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
    perimeter = 0
    for cell in board:
        perimeter += sum(1 for neighbor in neighbors4(cell) if neighbor not in board)
    ideal_perimeter = 2 * (math.sqrt(len(board)) + math.sqrt(len(board)))
    score += fill_ratio * 900.0
    score -= max(0.0, aspect - 1.8) * 160.0
    score -= max(0.0, perimeter - ideal_perimeter) * 1.7

    signatures = [canonical_signature(piece) for piece in pieces]
    duplicate_penalty = len(signatures) - len(set(signatures))
    score -= duplicate_penalty * 60.0
    skinny_penalty = 0
    for piece in pieces:
        min_px, min_py, max_px, max_py = bounds(piece)
        pw = max_px - min_px + 1
        ph = max_py - min_py + 1
        if max(pw, ph) >= 10 and min(pw, ph) <= 2:
            skinny_penalty += 1
    score -= skinny_penalty * 35.0
    return score


def analysis_meets_half_orientation_requirements(
    analysis: Analysis,
    args: argparse.Namespace,
) -> bool:
    return (
        analysis.horizontal_half_cell_count >= getattr(args, "min_horizontal_half_cells", 0)
        and analysis.vertical_half_cell_count >= getattr(args, "min_vertical_half_cells", 0)
        and analysis.horizontal_half_cell_contacts >= getattr(args, "min_horizontal_half_contacts", 0)
        and analysis.vertical_half_cell_contacts >= getattr(args, "min_vertical_half_contacts", 0)
    )


def generate_pair_swap_candidate(
    rng: random.Random,
    args: argparse.Namespace,
    attempt_index: int,
) -> Candidate | None:
    """Generate a chunky fixed-orientation multi-solution candidate.

    The idea is simple and very useful for physical tests: choose four robust
    shapes, use two copies of each shape, then pack those eight pieces into one
    connected board.  With piece artwork/IDs present, the player still has to
    decide which same-outline piece goes to which same-outline slot, and the
    fixed-orientation solver verifies that multiple complete assignments exist.
    """
    if not args.allow_identical_pieces:
        return None
    if args.pieces % 2 != 0 or args.pieces < 2:
        return None
    library = robust_pair_shape_library()
    pair_count = args.pieces // 2
    if len(library) < pair_count:
        return None

    board_w_small = args.board_w * SCALE
    board_h_small = args.board_h * SCALE
    base_indices = rng.sample(range(len(library)), pair_count)
    piece_indices = [index for index in base_indices for _ in range(2)]
    rng.shuffle(piece_indices)

    occupied: set[Cell] = set()
    for placed_count, shape_index in enumerate(piece_indices):
        shape = library[shape_index]
        _, _, max_shape_x, max_shape_y = bounds(shape)
        options: list[tuple[float, set[Cell]]] = []
        for ty in range(0, board_h_small - max_shape_y, SCALE):
            for tx in range(0, board_w_small - max_shape_x, SCALE):
                placed = {(x + tx, y + ty) for x, y in shape}
                if occupied & placed:
                    continue
                new_board = occupied | placed
                if not is_legal_half_cell_shape(new_board):
                    continue
                if placed_count > 0 and not any(
                    neighbor in occupied for cell in placed for neighbor in neighbors4(cell)
                ):
                    continue
                min_x, min_y, max_x, max_y = bounds(new_board)
                bbox_w = max_x - min_x + 1
                bbox_h = max_y - min_y + 1
                bbox_area = bbox_w * bbox_h
                fill_ratio = len(new_board) / bbox_area
                aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
                aspect_penalty = abs(aspect - 1.35)
                touch = sum(
                    1 for cell in placed for neighbor in neighbors4(cell) if neighbor in occupied
                )
                center_x = sum(x for x, _ in placed) / len(placed)
                center_y = sum(y for _, y in placed) / len(placed)
                center_bias = abs(center_x - board_w_small / 2) + abs(center_y - board_h_small / 2)
                score = (
                    bbox_area * 2.5
                    - fill_ratio * 45.0
                    + aspect_penalty * 18.0
                    - touch * 1.5
                    + center_bias * 0.02
                    + rng.random() * 0.2
                )
                options.append((score, placed))
        if not options:
            return None
        options.sort(key=lambda item: item[0])
        chosen = options[0][1]
        occupied |= chosen

    board = normalize_cells(occupied)
    pieces = [normalize_cells(library[index]) for index in piece_indices]
    board, pieces = prefer_landscape(board, pieces)
    if len(board) != args.pieces * PIECE_AREA_SMALL:
        return None
    if not is_connected(board):
        return None
    if not is_legal_half_cell_shape(board):
        return None
    if not args.allow_holes and has_small_hole(board):
        return None
    if has_quarter_artifact(board):
        return None
    if any(not validate_piece(piece, min_half_cells=args.min_half_cells) for piece in pieces):
        return None

    solution_count, solutions, placements_by_piece = count_solutions(
        board=board,
        pieces=pieces,
        allow_rotate=False,
        allow_mirror=False,
        limit=args.solution_count_limit,
    )
    if not (args.min_solutions <= solution_count <= args.max_solutions):
        return None

    rotated_solution_count, _, _ = count_solutions(
        board=board,
        pieces=pieces,
        allow_rotate=not args.no_rotate,
        allow_mirror=args.allow_mirror,
        limit=args.solution_count_limit,
    )
    analysis = analyze_candidate(
        board,
        pieces,
        solutions,
        solution_count,
        rotated_solution_count,
        placements_by_piece,
        args.min_solutions,
        args.max_solutions,
    )
    if analysis.quarter_artifact_count != 0 or analysis.fragile_artifact_count != 0:
        return None
    if analysis.duplicate_piece_count != 0 and not args.allow_identical_pieces:
        return None
    if not analysis_meets_half_orientation_requirements(analysis, args):
        return None
    return Candidate(
        board=board,
        pieces=pieces,
        solutions=solutions,
        solution_count=solution_count,
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=attempt_index,
    )


def generate_candidate(
    rng: random.Random,
    args: argparse.Namespace,
    attempt_index: int,
) -> Candidate | None:
    macro_area = args.pieces * (PIECE_AREA_SMALL // 4)
    macro_board = random_macro_board(
        rng,
        width=args.board_w,
        height=args.board_h,
        area=macro_area,
        allow_holes=args.allow_holes,
        max_tries=80,
    )
    if macro_board is None:
        return None
    regions = partition_into_tetrominoes(macro_board, args.pieces, rng)
    if regions is None:
        return None
    pieces = make_swapped_pieces(regions, rng, min_half_cells=args.min_half_cells)
    if pieces is None:
        return None
    board = macro_to_full_small(macro_board)
    board, pieces = normalize_board_and_pieces(board, pieces)
    board, pieces = prefer_landscape(board, pieces)

    if len(board) != args.pieces * PIECE_AREA_SMALL:
        return None
    if not is_connected(board):
        return None
    if not is_legal_half_cell_shape(board):
        return None
    if not args.allow_holes and has_small_hole(board):
        return None
    if has_quarter_artifact(board):
        return None
    if any(has_quarter_artifact(piece) for piece in pieces):
        return None
    if any(not validate_piece(piece, min_half_cells=args.min_half_cells) for piece in pieces):
        return None

    solution_count, solutions, placements_by_piece = count_solutions(
        board=board,
        pieces=pieces,
        allow_rotate=False,
        allow_mirror=False,
        limit=args.solution_count_limit,
    )
    rotated_solution_count, _, _ = count_solutions(
        board=board,
        pieces=pieces,
        allow_rotate=not args.no_rotate,
        allow_mirror=args.allow_mirror,
        limit=args.solution_count_limit,
    )
    if not (args.min_solutions <= solution_count <= args.max_solutions):
        return None
    analysis = analyze_candidate(
        board,
        pieces,
        solutions,
        solution_count,
        rotated_solution_count,
        placements_by_piece,
        args.min_solutions,
        args.max_solutions,
    )
    if analysis.quarter_artifact_count != 0 or analysis.fragile_artifact_count != 0:
        return None
    if analysis.duplicate_piece_count != 0 and not args.allow_identical_pieces:
        return None
    if not analysis_meets_half_orientation_requirements(analysis, args):
        return None
    return Candidate(
        board=board,
        pieces=pieces,
        solutions=solutions,
        solution_count=solution_count,
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=attempt_index,
    )


def guided_regions_from_super_layout(super_layout: list[MacroCell]) -> list[set[MacroCell]]:
    regions = []
    for sx, sy in super_layout:
        bx, by = sx * 2, sy * 2
        regions.append({(bx + dx, by + dy) for dx in range(2) for dy in range(2)})
    return regions


def generate_guided_candidate(
    args: argparse.Namespace,
    layout_index: int,
    swap_seed: int,
    attempt_index: int,
) -> Candidate | None:
    if args.pieces != 8:
        return None
    super_layout = GUIDED_SUPER_LAYOUTS[layout_index]
    max_super_x = max(x for x, _ in super_layout)
    max_super_y = max(y for _, y in super_layout)
    if (max_super_x + 1) * 2 > args.board_w or (max_super_y + 1) * 2 > args.board_h:
        return None

    regions = guided_regions_from_super_layout(super_layout)
    macro_board = {cell for region in regions for cell in region}
    if not args.allow_holes and has_macro_hole(macro_board):
        return None

    pieces = make_swapped_pieces(
        regions,
        random.Random(swap_seed),
        min_half_cells=args.min_half_cells,
        max_tries=1,
    )
    if pieces is None:
        return None

    signatures = [canonical_signature(piece) for piece in pieces]
    if len(set(signatures)) != len(signatures):
        return None

    board = macro_to_full_small(macro_board)
    board, pieces = normalize_board_and_pieces(board, pieces)
    board, pieces = prefer_landscape(board, pieces)
    if not is_connected(board):
        return None
    if not is_legal_half_cell_shape(board):
        return None
    if not args.allow_holes and has_small_hole(board):
        return None
    if any(not validate_piece(piece, min_half_cells=args.min_half_cells) for piece in pieces):
        return None

    solution_count, solutions, placements_by_piece = count_solutions(
        board=board,
        pieces=pieces,
        allow_rotate=False,
        allow_mirror=False,
        limit=args.solution_count_limit,
    )
    rotated_solution_count, _, _ = count_solutions(
        board=board,
        pieces=pieces,
        allow_rotate=not args.no_rotate,
        allow_mirror=args.allow_mirror,
        limit=args.solution_count_limit,
    )
    if not (args.min_solutions <= solution_count <= args.max_solutions):
        return None
    analysis = analyze_candidate(
        board,
        pieces,
        solutions,
        solution_count,
        rotated_solution_count,
        placements_by_piece,
        args.min_solutions,
        args.max_solutions,
    )
    if analysis.quarter_artifact_count != 0 or analysis.fragile_artifact_count != 0:
        return None
    if analysis.duplicate_piece_count != 0 and not args.allow_identical_pieces:
        return None
    if not analysis_meets_half_orientation_requirements(analysis, args):
        return None
    return Candidate(
        board=board,
        pieces=pieces,
        solutions=solutions,
        solution_count=solution_count,
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=attempt_index,
    )


def generate_guided_candidates(
    args: argparse.Namespace,
    start_time: float,
    max_count: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    if args.pieces != 8:
        return candidates

    # Seed 982 on layout 0 is a known good non-rectangular, non-duplicate
    # generated case for the default constraints.  The rest keep this as a
    # small guided search rather than a single hard-coded puzzle.
    seed_order = [982, 237, 124, 0, 1, 2, 3, 5, 8, 13, 21]
    known_seeds = set(seed_order)
    seed_order.extend(seed for seed in range(4_000) if seed not in known_seeds)
    attempt = 0
    seen_json: set[str] = set()
    for layout_index in range(len(GUIDED_SUPER_LAYOUTS)):
        for swap_seed in seed_order:
            if len(candidates) >= max_count:
                return candidates
            if time.monotonic() - start_time > args.time_limit:
                return candidates
            attempt += 1
            candidate = generate_guided_candidate(args, layout_index, swap_seed, attempt)
            if candidate is None:
                continue
            key = json.dumps(
                {
                    "board": sorted(candidate.board),
                    "pieces": [sorted(piece) for piece in candidate.pieces],
                }
            )
            if key in seen_json:
                continue
            seen_json.add(key)
            candidates.append(candidate)
            candidates.sort(key=lambda c: c.score, reverse=True)
            if args.verbose:
                print(
                    f"guided accepted #{len(candidates)} layout={layout_index} seed={swap_seed}: "
                    f"solutions={candidate.solution_count}, score={candidate.score:.1f}",
                    file=sys.stderr,
                )
    return candidates


def generate_candidates(args: argparse.Namespace) -> list[Candidate]:
    rng = random.Random(args.seed)
    start = time.monotonic()
    candidates: list[Candidate] = []
    seen_json: set[str] = set()
    attempts = 0
    pair_attempt_limit = max(args.candidates * 60, 360) if args.allow_identical_pieces else 0
    while attempts < pair_attempt_limit:
        attempts += 1
        if time.monotonic() - start > args.time_limit:
            break
        candidate = generate_pair_swap_candidate(rng, args, attempts)
        if candidate is None:
            if args.verbose and attempts % 25 == 0:
                print(f"pair attempts={attempts}, candidates={len(candidates)}", file=sys.stderr)
            continue
        key = json.dumps(
            {
                "board": sorted(candidate.board),
                "pieces": [sorted(piece) for piece in candidate.pieces],
            }
        )
        if key in seen_json:
            continue
        seen_json.add(key)
        candidates.append(candidate)
        candidates.sort(key=lambda c: c.score, reverse=True)
        if args.verbose:
            print(
                f"pair accepted #{len(candidates)} at attempt {attempts}: "
                f"fixed_solutions={candidate.solution_count}, score={candidate.score:.1f}",
                file=sys.stderr,
            )

    if len(candidates) < args.candidates and time.monotonic() - start <= args.time_limit:
        guided = generate_guided_candidates(args, start, args.candidates - len(candidates))
        for candidate in guided:
            key = json.dumps(
                {
                    "board": sorted(candidate.board),
                    "pieces": [sorted(piece) for piece in candidate.pieces],
                }
            )
            if key in seen_json:
                continue
            seen_json.add(key)
            candidates.append(candidate)
        candidates.sort(key=lambda c: c.score, reverse=True)

    while len(candidates) < args.candidates:
        attempts += 1
        if time.monotonic() - start > args.time_limit:
            break
        candidate = generate_candidate(rng, args, attempts)
        if candidate is None:
            if args.verbose and attempts % 25 == 0:
                print(f"attempts={attempts}, candidates={len(candidates)}", file=sys.stderr)
            continue
        key = json.dumps(
            {
                "board": sorted(candidate.board),
                "pieces": [sorted(piece) for piece in candidate.pieces],
            }
        )
        if key in seen_json:
            continue
        seen_json.add(key)
        candidates.append(candidate)
        candidates.sort(key=lambda c: c.score, reverse=True)
        if args.verbose:
            print(
                f"accepted #{len(candidates)} at attempt {attempts}: "
                f"solutions={candidate.solution_count}, score={candidate.score:.1f}",
                file=sys.stderr,
            )
    return candidates[: args.candidates]


def grid_string(
    cells: set[Cell] | frozenset[Cell],
    filled: str = "#",
    empty: str = ".",
    pad_to: tuple[int, int] | None = None,
) -> str:
    if not cells:
        return ""
    min_x, min_y, max_x, max_y = bounds(cells)
    if pad_to is not None:
        max_x = max(max_x, pad_to[0] - 1)
        max_y = max(max_y, pad_to[1] - 1)
        min_x = min(min_x, 0)
        min_y = min(min_y, 0)
    rows = []
    for y in range(min_y, max_y + 1):
        rows.append("".join(filled if (x, y) in cells else empty for x in range(min_x, max_x + 1)))
    return "\n".join(rows)


def solution_grid_string(board: set[Cell], solution: dict[int, frozenset[Cell]]) -> str:
    min_x, min_y, max_x, max_y = bounds(board)
    lookup: dict[Cell, str] = {}
    for piece_index, cells in solution.items():
        letter = LETTERS[piece_index % len(LETTERS)]
        for cell in cells:
            lookup[cell] = letter
    rows = []
    for y in range(min_y, max_y + 1):
        row = []
        for x in range(min_x, max_x + 1):
            if (x, y) not in board:
                row.append(".")
            else:
                row.append(lookup.get((x, y), "?"))
        rows.append("".join(row))
    return "\n".join(rows)


def candidate_to_text(candidate: Candidate, max_solutions: int = 2) -> str:
    lines = []
    lines.append("Board:")
    lines.append(grid_string(candidate.board))
    lines.append("")
    for i, solution in enumerate(candidate.solutions[:max_solutions], start=1):
        lines.append(f"Solution {i}:")
        lines.append(solution_grid_string(candidate.board, solution))
        lines.append("")
    for i, piece in enumerate(candidate.pieces):
        lines.append(f"Piece {LETTERS[i]}:")
        lines.append(grid_string(normalize_cells(piece)))
        lines.append("")
    lines.append("Analysis:")
    lines.append(json.dumps(analysis_to_json(candidate.analysis), ensure_ascii=False, indent=2))
    return "\n".join(lines)


def analysis_to_json(analysis: Analysis) -> dict[str, object]:
    return {
        "solution_count": analysis.solution_count,
        "rotated_solution_count": analysis.rotated_solution_count,
        "fixed_pieces": analysis.fixed_pieces,
        "movable_pieces": analysis.movable_pieces,
        "average_piece_candidates": analysis.average_piece_candidates,
        "half_cell_count_per_piece": analysis.half_cell_count_per_piece,
        "total_half_cell_count": analysis.total_half_cell_count,
        "horizontal_half_cell_count": analysis.horizontal_half_cell_count,
        "vertical_half_cell_count": analysis.vertical_half_cell_count,
        "horizontal_half_cell_count_per_piece": analysis.horizontal_half_cell_count_per_piece,
        "vertical_half_cell_count_per_piece": analysis.vertical_half_cell_count_per_piece,
        "horizontal_half_cell_contacts": analysis.horizontal_half_cell_contacts,
        "vertical_half_cell_contacts": analysis.vertical_half_cell_contacts,
        "quarter_artifact_count": analysis.quarter_artifact_count,
        "fragile_artifact_count": analysis.fragile_artifact_count,
        "duplicate_piece_count": analysis.duplicate_piece_count,
        "difficulty_score": analysis.difficulty_score,
    }


def candidate_to_json(candidate: Candidate) -> dict[str, object]:
    piece_areas_small = [len(piece) for piece in candidate.pieces]
    piece_areas_ordinary_equiv = [area / 4.0 for area in piece_areas_small]
    boundary_metrics = board_boundary_metrics(candidate.board)
    return {
        "scale": SCALE,
        "piece_count": len(candidate.pieces),
        "board": sorted([list(cell) for cell in candidate.board]),
        "board_boundary_metrics": boundary_metrics,
        "boundary_irregularities": boundary_metrics["boundary_irregularities"],
        "boundary_half_cell_irregularities": boundary_metrics["boundary_half_cell_irregularities"],
        "boundary_full_cell_irregularities": boundary_metrics["boundary_full_cell_irregularities"],
        "pieces": [
            {
                "id": LETTERS[i],
                "cells": sorted([list(cell) for cell in normalize_cells(piece)]),
                "area_small": len(piece),
                "area_ordinary_equiv": len(piece) / 4.0,
                "half_cell_count": count_half_cells(piece),
                "horizontal_half_cell_count": count_horizontal_half_cells(piece),
                "vertical_half_cell_count": count_vertical_half_cells(piece),
            }
            for i, piece in enumerate(candidate.pieces)
        ],
        "piece_areas_small": piece_areas_small,
        "piece_areas_ordinary_equiv": piece_areas_ordinary_equiv,
        "solution_count": candidate.solution_count,
        "rotated_solution_count": candidate.analysis.rotated_solution_count,
        "solutions": [
            {
                LETTERS[piece_index]: sorted([list(cell) for cell in cells])
                for piece_index, cells in sorted(solution.items())
            }
            for solution in candidate.solutions
        ],
        "score": candidate.score,
        "quarter_artifact_count": candidate.analysis.quarter_artifact_count,
        "fragile_artifact_count": candidate.analysis.fragile_artifact_count,
        "duplicate_piece_count": candidate.analysis.duplicate_piece_count,
        "horizontal_half_cell_count": candidate.analysis.horizontal_half_cell_count,
        "vertical_half_cell_count": candidate.analysis.vertical_half_cell_count,
        "horizontal_half_cell_count_per_piece": candidate.analysis.horizontal_half_cell_count_per_piece,
        "vertical_half_cell_count_per_piece": candidate.analysis.vertical_half_cell_count_per_piece,
        "horizontal_half_cell_contacts": candidate.analysis.horizontal_half_cell_contacts,
        "vertical_half_cell_contacts": candidate.analysis.vertical_half_cell_contacts,
        "analysis": analysis_to_json(candidate.analysis),
    }


def write_outputs(candidates: list[Candidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "candidates.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump([candidate_to_json(c) for c in candidates], f, ensure_ascii=False, indent=2)
    write_gallery_html(candidates, output_dir / "index.html")

    if not candidates:
        return

    for idx, candidate in enumerate(candidates, start=1):
        prefix = f"candidate_{idx:03d}"
        with (output_dir / f"{prefix}.txt").open("w", encoding="utf-8") as f:
            f.write(candidate_to_text(candidate))
        with (output_dir / f"{prefix}.json").open("w", encoding="utf-8") as f:
            json.dump(candidate_to_json(candidate), f, ensure_ascii=False, indent=2)
        write_candidate_svgs(candidate, output_dir, prefix)

    # Required convenient names for the best candidate.
    best = candidates[0]
    with (output_dir / "candidate.txt").open("w", encoding="utf-8") as f:
        f.write(candidate_to_text(best))
    with (output_dir / "candidate.json").open("w", encoding="utf-8") as f:
        json.dump(candidate_to_json(best), f, ensure_ascii=False, indent=2)
    write_candidate_svgs(best, output_dir, "candidate")


def write_gallery_html(candidates: list[Candidate], path: Path) -> None:
    data = json.dumps([candidate_to_json(c) for c in candidates], ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Half-cell puzzle gallery</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d4cfc4;
      --panel: #ffffff;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px 14px;
      border-bottom: 1px solid var(--line);
      background: #fffdf8;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary span, .metric {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 5px 8px;
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(260px, 34vw) 1fr;
      min-height: calc(100vh - 68px);
    }}
    #cards {{
      padding: 16px;
      overflow: auto;
      border-right: 1px solid var(--line);
      display: grid;
      align-content: start;
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      display: grid;
      gap: 8px;
    }}
    .card.active {{
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }}
    .card-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .card-title {{
      font-size: 15px;
      color: var(--ink);
      font-weight: 700;
    }}
    .mini {{
      width: 100%;
      max-height: 150px;
      display: block;
      border: 1px solid var(--line);
      background: #faf9f5;
    }}
    #detail {{
      padding: 18px;
      overflow: auto;
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .detail-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .detail-head h2 {{
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .section {{
      display: grid;
      gap: 8px;
    }}
    .section h3 {{
      margin: 0;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .viewer {{
      width: 100%;
      overflow: auto;
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 8px;
      padding: 10px;
    }}
    .solution-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .pieces {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 10px;
    }}
    .piece {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }}
    .piece-title {{
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    svg {{ max-width: 100%; height: auto; display: block; }}
    @media (max-width: 860px) {{
      header {{ align-items: start; flex-direction: column; }}
      main {{ grid-template-columns: 1fr; }}
      #cards {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        max-height: 48vh;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>半マスずれポリオミノ候補</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section id="cards"></section>
    <section id="detail"></section>
  </main>
  <script type="application/json" id="candidate-data">{data}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('candidate-data').textContent);
    const COLORS = {json.dumps(COLORS)};
    let selected = 0;

    function cellBounds(cells) {{
      let xs = cells.map(c => c[0]);
      let ys = cells.map(c => c[1]);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }}

    function key(c) {{ return c[0] + ',' + c[1]; }}

    function svgGrid(cells, assignment, labels, scale) {{
      if (!cells.length) return '';
      const [minX, minY, maxX, maxY] = cellBounds(cells);
      const size = scale || 14;
      const pad = 10;
      const w = (maxX - minX + 1) * size + pad * 2;
      const h = (maxY - minY + 1) * size + pad * 2;
      const cellSet = new Set(cells.map(key));
      let rects = '';
      for (const c of cells) {{
        const k = key(c);
        const p = assignment ? assignment[k] : -1;
        const fill = p >= 0 ? COLORS[p % COLORS.length] : '#cfd8dc';
        const x = pad + (c[0] - minX) * size;
        const y = pad + (c[1] - minY) * size;
        rects += `<rect x="${{x}}" y="${{y}}" width="${{size}}" height="${{size}}" fill="${{fill}}"/>`;
      }}
      let lines = '';
      for (let x = minX; x <= maxX + 1; x++) {{
        const sx = pad + (x - minX) * size;
        const strong = x % 2 === 0;
        lines += `<line x1="${{sx}}" y1="${{pad}}" x2="${{sx}}" y2="${{pad + (maxY - minY + 1) * size}}" stroke="${{strong ? '#817c70' : '#d7d2c6'}}" stroke-width="${{strong ? 1.4 : 0.7}}"/>`;
      }}
      for (let y = minY; y <= maxY + 1; y++) {{
        const sy = pad + (y - minY) * size;
        const strong = y % 2 === 0;
        lines += `<line x1="${{pad}}" y1="${{sy}}" x2="${{pad + (maxX - minX + 1) * size}}" y2="${{sy}}" stroke="${{strong ? '#817c70' : '#d7d2c6'}}" stroke-width="${{strong ? 1.4 : 0.7}}"/>`;
      }}
      let borders = '';
      if (assignment) {{
        const dirs = [[-1,0,'l'], [1,0,'r'], [0,-1,'t'], [0,1,'b']];
        for (const c of cells) {{
          const p = assignment[key(c)];
          const x = pad + (c[0] - minX) * size;
          const y = pad + (c[1] - minY) * size;
          for (const [dx, dy, side] of dirs) {{
            const nk = (c[0] + dx) + ',' + (c[1] + dy);
            if (cellSet.has(nk) && assignment[nk] === p) continue;
            if (side === 'l') borders += `<line x1="${{x}}" y1="${{y}}" x2="${{x}}" y2="${{y + size}}" stroke="#222" stroke-width="2"/>`;
            if (side === 'r') borders += `<line x1="${{x + size}}" y1="${{y}}" x2="${{x + size}}" y2="${{y + size}}" stroke="#222" stroke-width="2"/>`;
            if (side === 't') borders += `<line x1="${{x}}" y1="${{y}}" x2="${{x + size}}" y2="${{y}}" stroke="#222" stroke-width="2"/>`;
            if (side === 'b') borders += `<line x1="${{x}}" y1="${{y + size}}" x2="${{x + size}}" y2="${{y + size}}" stroke="#222" stroke-width="2"/>`;
          }}
        }}
      }}
      let text = '';
      if (labels && assignment) {{
        for (const label of Object.keys(labels)) {{
          const pts = labels[label];
          const cx = pts.reduce((a, c) => a + c[0], 0) / pts.length;
          const cy = pts.reduce((a, c) => a + c[1], 0) / pts.length;
          const sx = pad + (cx - minX + 0.5) * size;
          const sy = pad + (cy - minY + 0.5) * size;
          text += `<text x="${{sx}}" y="${{sy}}" text-anchor="middle" dominant-baseline="central" font-size="${{Math.max(10, size)}}" font-weight="700" font-family="Arial" fill="#202020" paint-order="stroke" stroke="#fff" stroke-width="3">${{label}}</text>`;
        }}
      }}
      return `<svg viewBox="0 0 ${{w}} ${{h}}" role="img" aria-label="candidate grid">${{rects}}${{borders}}${{lines}}${{text}}</svg>`;
    }}

    function assignmentFromSolution(solution) {{
      const assignment = {{}};
      const labels = {{}};
      Object.keys(solution).forEach((id, idx) => {{
        labels[id] = solution[id];
        for (const c of solution[id]) assignment[key(c)] = idx;
      }});
      return [assignment, labels];
    }}

    function renderCards() {{
      const cards = document.getElementById('cards');
      cards.innerHTML = DATA.map((c, i) => `
        <article class="card ${{i === selected ? 'active' : ''}}" data-i="${{i}}">
          <div class="card-top"><span class="card-title">候補 ${{i + 1}}</span><span>固定 ${{c.solution_count}} 解</span></div>
          <div class="mini">${{svgGrid(c.board, null, null, 7)}}</div>
          <div class="metrics">
            <span class="metric">脆さ ${{c.fragile_artifact_count}}</span>
            <span class="metric">重複 ${{c.duplicate_piece_count}}</span>
            <span class="metric">1/4 ${{c.quarter_artifact_count}}</span>
            <span class="metric">半マス ${{c.analysis.total_half_cell_count}}</span>
            <span class="metric">横半 ${{c.horizontal_half_cell_count}}</span>
            <span class="metric">縦半 ${{c.vertical_half_cell_count}}</span>
            <span class="metric">横接触 ${{c.horizontal_half_cell_contacts}}</span>
            <span class="metric">縦接触 ${{c.vertical_half_cell_contacts}}</span>
            <span class="metric">面積 ${{c.piece_areas_ordinary_equiv.map(v => v.toFixed(1)).join(',')}}</span>
          </div>
        </article>
      `).join('');
      cards.querySelectorAll('.card').forEach(card => {{
        card.addEventListener('click', () => {{
          selected = Number(card.dataset.i);
          render();
        }});
      }});
    }}

    function renderDetail() {{
      const c = DATA[selected];
      const detail = document.getElementById('detail');
      const solutions = c.solutions.slice(0, 8).map((s, i) => {{
        const [a, labels] = assignmentFromSolution(s);
        return `<div class="viewer"><h3>Solution ${{i + 1}}</h3>${{svgGrid(c.board, a, labels, 16)}}</div>`;
      }}).join('');
      const pieces = c.pieces.map((p, i) => `
        <div class="piece">
          <div class="piece-title">Piece ${{p.id}} / 面積 ${{p.area_small}} (${{p.area_ordinary_equiv.toFixed(1)}}) / 半マス ${{p.half_cell_count}} / 横 ${{p.horizontal_half_cell_count}} / 縦 ${{p.vertical_half_cell_count}}</div>
          ${{svgGrid(p.cells, Object.fromEntries(p.cells.map(cell => [key(cell), i])), null, 14)}}
        </div>
      `).join('');
      detail.innerHTML = `
        <div class="detail-head">
          <h2>候補 ${{selected + 1}}</h2>
          <div class="metrics">
            <span class="metric">固定向き ${{c.solution_count}} 解</span>
            <span class="metric">回転あり ${{c.rotated_solution_count}} 解</span>
            <span class="metric">脆さ ${{c.fragile_artifact_count}}</span>
            <span class="metric">重複 ${{c.duplicate_piece_count}}</span>
            <span class="metric">1/4 ${{c.quarter_artifact_count}}</span>
            <span class="metric">横半 ${{c.horizontal_half_cell_count}}</span>
            <span class="metric">縦半 ${{c.vertical_half_cell_count}}</span>
            <span class="metric">横半/ピース ${{c.horizontal_half_cell_count_per_piece.join(',')}}</span>
            <span class="metric">縦半/ピース ${{c.vertical_half_cell_count_per_piece.join(',')}}</span>
            <span class="metric">横接触 ${{c.horizontal_half_cell_contacts}}</span>
            <span class="metric">縦接触 ${{c.vertical_half_cell_contacts}}</span>
            <span class="metric">面積 small ${{c.piece_areas_small.join(',')}}</span>
            <span class="metric">面積 ordinary ${{c.piece_areas_ordinary_equiv.map(v => v.toFixed(1)).join(',')}}</span>
            <span class="metric">Score ${{c.score.toFixed(1)}}</span>
          </div>
        </div>
        <div class="section">
          <h3>Board</h3>
          <div class="viewer">${{svgGrid(c.board, null, null, 16)}}</div>
        </div>
        <div class="section">
          <h3>Solutions</h3>
          <div class="solution-grid">${{solutions}}</div>
        </div>
        <div class="section">
          <h3>Pieces</h3>
          <div class="pieces">${{pieces}}</div>
        </div>
      `;
    }}

    function renderSummary() {{
      document.getElementById('summary').innerHTML = `
        <span>${{DATA.length}} candidates</span>
        <span>fixed orientation</span>
        <span>fragile = 0</span>
        <span>duplicate = 0</span>
        <span>quarter = 0</span>
      `;
    }}

    function render() {{
      renderSummary();
      renderCards();
      renderDetail();
    }}
    render();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_candidate_svgs(candidate: Candidate, output_dir: Path, prefix: str) -> None:
    (output_dir / f"{prefix}_board.svg").write_text(
        svg_board(candidate.board), encoding="utf-8"
    )
    (output_dir / f"{prefix}_pieces.svg").write_text(
        svg_pieces(candidate.pieces), encoding="utf-8"
    )
    for i, solution in enumerate(candidate.solutions[:2], start=1):
        (output_dir / f"{prefix}_solution_{i}.svg").write_text(
            svg_solution(candidate.board, solution), encoding="utf-8"
        )


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#f8f7f2"/>\n'
    )


def svg_grid(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    cell_size: int,
    ox: int,
    oy: int,
) -> str:
    lines = []
    width = (max_x - min_x + 1) * cell_size
    height = (max_y - min_y + 1) * cell_size
    for x in range(min_x, max_x + 2):
        sx = ox + (x - min_x) * cell_size
        stroke = "#817c70" if x % SCALE == 0 else "#d7d2c6"
        sw = 1.6 if x % SCALE == 0 else 0.8
        lines.append(
            f'<line x1="{sx}" y1="{oy}" x2="{sx}" y2="{oy + height}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
    for y in range(min_y, max_y + 2):
        sy = oy + (y - min_y) * cell_size
        stroke = "#817c70" if y % SCALE == 0 else "#d7d2c6"
        sw = 1.6 if y % SCALE == 0 else 0.8
        lines.append(
            f'<line x1="{ox}" y1="{sy}" x2="{ox + width}" y2="{sy}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
    return "\n".join(lines)


def svg_board(board: set[Cell]) -> str:
    min_x, min_y, max_x, max_y = bounds(board)
    cell_size = 18
    margin = 28
    width = (max_x - min_x + 1) * cell_size + margin * 2
    height = (max_y - min_y + 1) * cell_size + margin * 2
    parts = [svg_header(width, height)]
    for x, y in sorted(board):
        sx = margin + (x - min_x) * cell_size
        sy = margin + (y - min_y) * cell_size
        parts.append(
            f'<rect x="{sx}" y="{sy}" width="{cell_size}" height="{cell_size}" '
            'fill="#cfd8dc" stroke="none"/>'
        )
    parts.append(svg_grid(min_x, min_y, max_x, max_y, cell_size, margin, margin))
    parts.append("</svg>\n")
    return "\n".join(parts)


def svg_solution(board: set[Cell], solution: dict[int, frozenset[Cell]]) -> str:
    min_x, min_y, max_x, max_y = bounds(board)
    cell_size = 18
    margin = 28
    width = (max_x - min_x + 1) * cell_size + margin * 2
    height = (max_y - min_y + 1) * cell_size + margin * 2
    lookup: dict[Cell, int] = {}
    for piece_index, cells in solution.items():
        for cell in cells:
            lookup[cell] = piece_index

    parts = [svg_header(width, height)]
    for cell, piece_index in sorted(lookup.items()):
        x, y = cell
        sx = margin + (x - min_x) * cell_size
        sy = margin + (y - min_y) * cell_size
        color = COLORS[piece_index % len(COLORS)]
        parts.append(
            f'<rect x="{sx}" y="{sy}" width="{cell_size}" height="{cell_size}" '
            f'fill="{color}" stroke="none"/>'
        )

    # Thick boundaries where adjacent small cells belong to different pieces.
    for x, y in sorted(board):
        piece_index = lookup.get((x, y))
        sx = margin + (x - min_x) * cell_size
        sy = margin + (y - min_y) * cell_size
        for edge, neighbor in (
            ("left", (x - 1, y)),
            ("right", (x + 1, y)),
            ("top", (x, y - 1)),
            ("bottom", (x, y + 1)),
        ):
            if neighbor in board and lookup.get(neighbor) == piece_index:
                continue
            if edge == "left":
                parts.append(boundary_line(sx, sy, sx, sy + cell_size))
            elif edge == "right":
                parts.append(boundary_line(sx + cell_size, sy, sx + cell_size, sy + cell_size))
            elif edge == "top":
                parts.append(boundary_line(sx, sy, sx + cell_size, sy))
            else:
                parts.append(boundary_line(sx, sy + cell_size, sx + cell_size, sy + cell_size))

    parts.append(svg_grid(min_x, min_y, max_x, max_y, cell_size, margin, margin))
    for piece_index, cells in solution.items():
        cx = sum(x for x, _ in cells) / len(cells)
        cy = sum(y for _, y in cells) / len(cells)
        sx = margin + (cx - min_x + 0.5) * cell_size
        sy = margin + (cy - min_y + 0.5) * cell_size
        parts.append(
            f'<text x="{sx:.1f}" y="{sy:.1f}" text-anchor="middle" dominant-baseline="central" '
            'font-family="Arial, sans-serif" font-size="16" font-weight="700" '
            'fill="#202020" paint-order="stroke" stroke="#ffffff" stroke-width="3">'
            f"{LETTERS[piece_index]}</text>"
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def boundary_line(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        'stroke="#1f1f1f" stroke-width="2.3" stroke-linecap="square"/>'
    )


def svg_pieces(pieces: list[set[Cell]]) -> str:
    cell_size = 18
    margin = 28
    gap_x = 42
    gap_y = 48
    panels = []
    panel_widths = []
    panel_heights = []
    for i, piece in enumerate(pieces):
        piece = normalize_cells(piece)
        min_x, min_y, max_x, max_y = bounds(piece)
        w = (max_x - min_x + 1) * cell_size
        h = (max_y - min_y + 1) * cell_size
        panel_widths.append(max(w, 72))
        panel_heights.append(h + 22)
        panels.append((i, piece, min_x, min_y, max_x, max_y, w, h))

    cols = min(4, len(pieces))
    rows = math.ceil(len(pieces) / cols)
    col_width = max(panel_widths) + gap_x
    row_height = max(panel_heights) + gap_y
    width = margin * 2 + cols * col_width - gap_x
    height = margin * 2 + rows * row_height - gap_y
    parts = [svg_header(width, height)]

    for i, piece, min_x, min_y, max_x, max_y, _, _ in panels:
        col = i % cols
        row = i // cols
        ox = margin + col * col_width
        oy = margin + row * row_height + 22
        parts.append(
            f'<text x="{ox}" y="{oy - 8}" font-family="Arial, sans-serif" '
            'font-size="15" font-weight="700" fill="#202020">'
            f"Piece {LETTERS[i]}</text>"
        )
        for x, y in sorted(piece):
            sx = ox + (x - min_x) * cell_size
            sy = oy + (y - min_y) * cell_size
            color = COLORS[i % len(COLORS)]
            parts.append(
                f'<rect x="{sx}" y="{sy}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color}" stroke="none"/>'
            )
        for x, y in sorted(piece):
            sx = ox + (x - min_x) * cell_size
            sy = oy + (y - min_y) * cell_size
            for edge, neighbor in (
                ("left", (x - 1, y)),
                ("right", (x + 1, y)),
                ("top", (x, y - 1)),
                ("bottom", (x, y + 1)),
            ):
                if neighbor in piece:
                    continue
                if edge == "left":
                    parts.append(boundary_line(sx, sy, sx, sy + cell_size))
                elif edge == "right":
                    parts.append(boundary_line(sx + cell_size, sy, sx + cell_size, sy + cell_size))
                elif edge == "top":
                    parts.append(boundary_line(sx, sy, sx + cell_size, sy))
                else:
                    parts.append(boundary_line(sx, sy + cell_size, sx + cell_size, sy + cell_size))
        parts.append(svg_grid(min_x, min_y, max_x, max_y, cell_size, ox, oy))

    parts.append("</svg>\n")
    return "\n".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate half-cell polyomino-style tiling puzzle candidates."
    )
    parser.add_argument("--pieces", type=int, default=PIECE_COUNT)
    parser.add_argument("--board-w", type=int, default=BOARD_W)
    parser.add_argument("--board-h", type=int, default=BOARD_H)
    parser.add_argument("--min-solutions", type=int, default=TARGET_MIN_SOLUTIONS)
    parser.add_argument("--max-solutions", type=int, default=TARGET_MAX_SOLUTIONS)
    parser.add_argument("--allow-mirror", action="store_true", default=ALLOW_MIRROR)
    parser.add_argument("--no-rotate", action="store_true", default=not ALLOW_ROTATE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--allow-holes", action="store_true")
    parser.add_argument("--allow-identical-pieces", action="store_true")
    parser.add_argument("--min-half-cells", type=int, default=0)
    parser.add_argument("--min-horizontal-half-cells", type=int, default=1)
    parser.add_argument("--min-vertical-half-cells", type=int, default=1)
    parser.add_argument("--min-horizontal-half-contacts", type=int, default=0)
    parser.add_argument("--min-vertical-half-contacts", type=int, default=0)
    parser.add_argument("--solution-count-limit", type=int, default=SOLUTION_COUNT_LIMIT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.pieces <= 0:
        raise SystemExit("--pieces must be positive")
    if args.pieces > len(LETTERS):
        raise SystemExit(f"--pieces must be at most {len(LETTERS)}")
    if args.board_w <= 0 or args.board_h <= 0:
        raise SystemExit("--board-w and --board-h must be positive")
    macro_area = args.pieces * (PIECE_AREA_SMALL // 4)
    if macro_area > args.board_w * args.board_h:
        raise SystemExit("board is too small for the requested piece count")
    if args.min_solutions > args.max_solutions:
        raise SystemExit("--min-solutions cannot exceed --max-solutions")
    if args.solution_count_limit < args.max_solutions:
        raise SystemExit("--solution-count-limit must be at least --max-solutions")
    if args.min_half_cells < 0:
        raise SystemExit("--min-half-cells must be non-negative")
    if args.min_horizontal_half_cells < 0 or args.min_vertical_half_cells < 0:
        raise SystemExit("directional half-cell minimums must be non-negative")
    if args.min_horizontal_half_contacts < 0 or args.min_vertical_half_contacts < 0:
        raise SystemExit("directional half-cell contact minimums must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    candidates = generate_candidates(args)
    write_outputs(candidates, args.output_dir)
    if not candidates:
        print(
            "No candidate found. Try increasing --time-limit, changing --seed, "
            "or lowering --min-solutions.",
            file=sys.stderr,
        )
        return 1

    best = candidates[0]
    print(candidate_to_text(best, max_solutions=2))
    print(f"Saved {len(candidates)} candidate(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
