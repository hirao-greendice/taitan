#!/usr/bin/env python3
"""Heavy search for ideal half-cell puzzle candidates.

This script is intentionally stricter and heavier than generate_half_polyomino.py.
It searches for six non-identical pieces that tile a near-rectangular/full
rectangular board in at least K fixed-orientation ways.

Pipeline:
  1. Build many candidate piece shapes by generating legal tilings of a full
     rectangular board, then adding single-cell half transfers across region
     boundaries.  This produces pieces that are known to be practical tiling
     fragments rather than random disconnected-looking shapes.
  2. Enumerate every fixed-orientation placement of every candidate piece.
  3. Use OR-Tools CP-SAT to select six distinct pieces and K distinct exact
     covers of the same board.
  4. Verify the selected set with the existing exact-cover counter and write the
     normal JSON/SVG/HTML outputs.

Install:
  python -m pip install -r requirements.txt

Example:
  python ideal_half_polyomino_search.py --time-limit 1800 --library-target 2500 --output-dir out_ideal
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from ortools.sat.python import cp_model
except ModuleNotFoundError:  # pragma: no cover - user environment guard
    cp_model = None

import generate_half_polyomino as base


Cell = tuple[int, int]
MacroCell = tuple[int, int]


@dataclass(frozen=True)
class ShapeRecord:
    id: int
    cells: frozenset[Cell]
    area: int
    half_count: int
    horizontal_half_count: int
    vertical_half_count: int
    fragile_count: int
    rotational_signature: tuple[Cell, ...]


@dataclass(frozen=True)
class PlacementRecord:
    id: int
    shape_id: int
    cells: frozenset[Cell]


@dataclass(frozen=True)
class CpsatSolutionProof:
    layer: int
    placements: tuple[PlacementRecord, ...]


@dataclass(frozen=True)
class CpsatCandidate:
    shapes: tuple[ShapeRecord, ...]
    proofs: tuple[CpsatSolutionProof, ...]


def align_cells(cells: set[Cell] | frozenset[Cell]) -> set[Cell]:
    """Normalize without shifting the 2x2 ordinary-cell grid out of phase."""
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    shift_x = min_x - (min_x % base.SCALE)
    shift_y = min_y - (min_y % base.SCALE)
    return {(x - shift_x, y - shift_y) for x, y in cells}


def exact_signature(cells: set[Cell] | frozenset[Cell]) -> tuple[Cell, ...]:
    return tuple(sorted(align_cells(cells)))


def rotational_signature(cells: set[Cell] | frozenset[Cell]) -> tuple[Cell, ...]:
    variants = []
    for rotation in range(4):
        oriented = align_cells(base.transform_cells(cells, rotation, False))
        if base.is_legal_half_cell_shape(oriented):
            variants.append(tuple(sorted(oriented)))
    if not variants:
        return tuple(sorted(align_cells(cells)))
    return min(variants)


def rectangle_board(width_small: int, height_small: int) -> set[Cell]:
    return {(x, y) for y in range(height_small) for x in range(width_small)}


def macro_rectangle(width_macro: int, height_macro: int) -> set[MacroCell]:
    return {(x, y) for y in range(height_macro) for x in range(width_macro)}


def macro_to_small_board(cells: set[MacroCell]) -> set[Cell]:
    return base.macro_to_full_small(cells)


def half_notch_options(cells: set[MacroCell]) -> list[tuple[MacroCell, int]]:
    options: list[tuple[MacroCell, int]] = []
    for x, y in sorted(cells):
        checks = (
            ((x - 1, y), base.MASK_RIGHT),
            ((x + 1, y), base.MASK_LEFT),
            ((x, y - 1), base.MASK_BOTTOM),
            ((x, y + 1), base.MASK_TOP),
        )
        for neighbor, keep_mask in checks:
            if neighbor not in cells:
                options.append(((x, y), keep_mask))
    return options


def macro_to_small_board_with_half_notches(
    cells: set[MacroCell],
    notches: tuple[tuple[MacroCell, int], ...],
) -> set[Cell]:
    masks = {cell: base.MASK_FULL for cell in cells}
    for cell, keep_mask in notches:
        masks[cell] = keep_mask
    return base.masks_to_cells(masks)


def generate_boundary_small_boards(
    args: argparse.Namespace,
    macro_board: set[MacroCell],
) -> list[set[Cell]]:
    min_irregularities = int(getattr(args, "min_boundary_irregularities", 0))
    min_half_notches = int(getattr(args, "min_boundary_half_notches", 0))
    max_half_notches = int(getattr(args, "max_boundary_half_notches", max(2, min_half_notches)))
    max_variants = int(getattr(args, "max_board_variants_per_macro", 1))
    allow_holes = bool(getattr(args, "allow_holes", False))
    candidates: list[set[Cell]] = []
    seen: set[tuple[Cell, ...]] = set()

    def maybe_add(board: set[Cell]) -> None:
        if not board:
            return
        signature = tuple(sorted(board))
        if signature in seen:
            return
        seen.add(signature)
        if not base.is_connected(board):
            return
        if not base.is_legal_half_cell_shape(board):
            return
        if not allow_holes and base.has_small_hole(board):
            return
        metrics = base.board_boundary_metrics(board)
        if metrics["boundary_irregularities"] < min_irregularities:
            return
        if metrics["boundary_half_cell_irregularities"] < min_half_notches:
            return
        if not (args.pieces * args.min_piece_area <= len(board) <= args.pieces * args.max_piece_area):
            return
        candidates.append(board)

    maybe_add(macro_to_small_board(macro_board))
    options = half_notch_options(macro_board)
    for notch_count in range(max(0, min_half_notches), max_half_notches + 1):
        if notch_count == 0:
            continue
        for combo_index, combo in enumerate(itertools.combinations(options, notch_count)):
            if combo_index >= args.max_board_candidates_per_remove:
                break
            cells = [cell for cell, _ in combo]
            if len(set(cells)) != len(cells):
                continue
            maybe_add(macro_to_small_board_with_half_notches(macro_board, combo))

    candidates.sort(
        key=lambda board: (
            base.board_boundary_metrics(board)["boundary_half_cell_irregularities"],
            base.board_boundary_metrics(board)["boundary_irregularities"],
            -abs(len(board) - args.pieces * ((args.min_piece_area + args.max_piece_area) / 2)),
        ),
        reverse=True,
    )
    return candidates[:max_variants]


def macro_perimeter(cells: set[MacroCell]) -> int:
    return sum(1 for cell in cells for neighbor in base.neighbors4(cell) if neighbor not in cells)


def narrow_corridor_penalty(cells: set[MacroCell]) -> int:
    horizontal: set[MacroCell] = set()
    vertical: set[MacroCell] = set()
    for x, y in cells:
        left = (x - 1, y) in cells
        right = (x + 1, y) in cells
        up = (x, y - 1) in cells
        down = (x, y + 1) in cells
        if left and right and not up and not down:
            horizontal.add((x, y))
        if up and down and not left and not right:
            vertical.add((x, y))

    penalty = 0
    for run_cells, axis in ((horizontal, "x"), (vertical, "y")):
        remaining = set(run_cells)
        while remaining:
            start = remaining.pop()
            stack = [start]
            run = {start}
            while stack:
                x, y = stack.pop()
                neighbors = (
                    ((x - 1, y), (x + 1, y))
                    if axis == "x"
                    else ((x, y - 1), (x, y + 1))
                )
                for neighbor in neighbors:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        run.add(neighbor)
                        stack.append(neighbor)
            penalty += max(0, len(run) - 2)
    return penalty


def board_score(
    cells: set[MacroCell],
    base_width: int,
    base_height: int,
    roughness: str = "balanced",
) -> float:
    min_x = min(x for x, _ in cells)
    max_x = max(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_y = max(y for _, y in cells)
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    fill = len(cells) / (bbox_w * bbox_h)
    removed = base_width * base_height - len(cells)
    perimeter = macro_perimeter(cells)
    ideal_perimeter = 2 * (bbox_w + bbox_h)
    aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
    weights = {
        "neat": (1100.0, 25.0, 24.0, 1.65, 130.0, 12.0),
        "balanced": (1000.0, 25.0, 20.0, 1.8, 100.0, 8.0),
        "rough": (420.0, 12.0, 6.0, 2.25, 35.0, 4.0),
        "wild": (180.0, 6.0, 2.0, 3.0, 15.0, 2.0),
    }
    fill_weight, remove_weight, perimeter_weight, aspect_limit, aspect_weight, corridor_weight = weights.get(
        roughness,
        weights["balanced"],
    )
    return (
        fill * fill_weight
        - removed * remove_weight
        - max(0, perimeter - ideal_perimeter) * perimeter_weight
        - max(0.0, aspect - aspect_limit) * aspect_weight
        - narrow_corridor_penalty(cells) * corridor_weight
    )


def boundary_macro_cells(cells: set[MacroCell]) -> list[MacroCell]:
    return sorted(
        cell
        for cell in cells
        if any(neighbor not in cells for neighbor in base.neighbors4(cell))
    )


def generate_near_rect_macro_boards(args: argparse.Namespace) -> list[set[MacroCell]]:
    base_board = macro_rectangle(args.board_w_macro, args.board_h_macro)
    candidates: list[set[MacroCell]] = []
    seen: set[tuple[MacroCell, ...]] = set()
    boundary = boundary_macro_cells(base_board)
    roughness = getattr(args, "board_roughness", "balanced")

    for remove_count in range(args.board_remove_min, args.board_remove_max + 1):
        if remove_count == 0:
            boards = [set(base_board)]
        else:
            combos = itertools.combinations(boundary, remove_count)
            boards = []
            for combo_index, combo in enumerate(combos):
                if combo_index >= args.max_board_candidates_per_remove:
                    break
                boards.append(base_board - set(combo))

        for board in boards:
            if len(board) == 0:
                continue
            signature = tuple(sorted(board))
            if signature in seen:
                continue
            seen.add(signature)
            if not base.is_macro_connected(board):
                continue
            if not args.allow_holes and base.has_macro_hole(board):
                continue
            small_area = len(board) * 4
            if not (args.pieces * args.min_piece_area <= small_area <= args.pieces * args.max_piece_area):
                continue
            candidates.append(board)

    candidates.sort(
        key=lambda board: board_score(board, args.board_w_macro, args.board_h_macro, roughness),
        reverse=True,
    )
    return candidates[: args.max_board_candidates]


def complement_half(mask: int) -> int:
    if mask == base.MASK_TOP:
        return base.MASK_BOTTOM
    if mask == base.MASK_BOTTOM:
        return base.MASK_TOP
    if mask == base.MASK_LEFT:
        return base.MASK_RIGHT
    if mask == base.MASK_RIGHT:
        return base.MASK_LEFT
    raise ValueError(f"not a half mask: {mask}")


def transfer_masks(direction: str, receiver_on_positive_side: bool) -> tuple[int, int]:
    """Return (donor_mask, receiver_mask) for one split ordinary cell."""
    if direction == "h":
        # Neighbor is horizontally adjacent.  Receiver gets the side touching it.
        receiver = base.MASK_RIGHT if receiver_on_positive_side else base.MASK_LEFT
    else:
        receiver = base.MASK_BOTTOM if receiver_on_positive_side else base.MASK_TOP
    return complement_half(receiver), receiver


def choose_area_pattern(
    rng: random.Random,
    piece_count: int,
    total_area: int,
    min_area: int,
    max_area: int,
) -> list[int]:
    allowed = [area for area in range(min_area, max_area + 1, 2)]
    for _ in range(10_000):
        areas = [rng.choice(allowed) for _ in range(piece_count - 1)]
        last = total_area - sum(areas)
        if last in allowed:
            areas.append(last)
            rng.shuffle(areas)
            return areas
    raise RuntimeError("could not create area pattern")


def choose_macro_region_sizes(
    rng: random.Random,
    board_area_macro: int,
    piece_count: int,
) -> list[int] | None:
    """Choose connected macro-region sizes before half-cell transfers.

    Region sizes of 3, 4, and 5 ordinary cells are useful because half transfers
    can turn them into 14..18 small-cell pieces.  This also supports boards such
    as 5x5 ordinary cells, whose area is not divisible by 4.
    """
    for _ in range(10_000):
        sizes = [rng.choice((3, 4, 5)) for _ in range(piece_count - 1)]
        last = board_area_macro - sum(sizes)
        if last in (3, 4, 5):
            sizes.append(last)
            rng.shuffle(sizes)
            return sizes
    return None


@lru_cache(maxsize=200_000)
def _connected_subsets_containing_cached(
    remaining_signature: tuple[MacroCell, ...],
    anchor: MacroCell,
    size: int,
    limit: int,
) -> tuple[tuple[MacroCell, ...], ...]:
    remaining = set(remaining_signature)
    subsets: set[frozenset[MacroCell]] = set()
    stack: list[tuple[frozenset[MacroCell], frozenset[MacroCell]]] = [
        (frozenset({anchor}), frozenset(n for n in base.neighbors4(anchor) if n in remaining))
    ]
    while stack and len(subsets) < limit:
        shape, frontier = stack.pop()
        if len(shape) == size:
            subsets.add(shape)
            continue
        for cell in list(frontier):
            new_shape = set(shape)
            new_shape.add(cell)
            new_frontier = set(frontier)
            new_frontier.remove(cell)
            for neighbor in base.neighbors4(cell):
                if neighbor in remaining and neighbor not in new_shape:
                    new_frontier.add(neighbor)
            stack.append((frozenset(new_shape), frozenset(new_frontier)))
    return tuple(tuple(sorted(subset)) for subset in subsets)


def connected_subsets_containing(
    remaining: set[MacroCell],
    anchor: MacroCell,
    size: int,
    limit: int = 400,
) -> list[frozenset[MacroCell]]:
    return [
        frozenset(subset)
        for subset in _connected_subsets_containing_cached(
            tuple(sorted(remaining)), anchor, size, limit
        )
    ]


def partition_macro_board(
    rng: random.Random,
    board: set[MacroCell],
    piece_count: int,
    max_nodes: int = 20_000,
) -> list[set[MacroCell]] | None:
    sizes = choose_macro_region_sizes(rng, len(board), piece_count)
    if sizes is None:
        return None
    sizes.sort(reverse=True)
    nodes = 0

    def search(
        remaining: frozenset[MacroCell],
        remaining_sizes: tuple[int, ...],
    ) -> list[set[MacroCell]] | None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return None
        if not remaining_sizes:
            return [] if not remaining else None
        if sum(remaining_sizes) != len(remaining):
            return None

        size = remaining_sizes[0]
        anchor = min(remaining)
        options = connected_subsets_containing(set(remaining), anchor, size)
        rng.shuffle(options)
        for subset in options:
            rest = frozenset(set(remaining) - set(subset))
            if rest and not base.is_macro_connected(set(rest)):
                # A disconnected remainder can never be fully covered by
                # connected regions without crossing occupied cells.
                continue
            result = search(rest, remaining_sizes[1:])
            if result is not None:
                return [set(subset)] + result
        return None

    return search(frozenset(board), tuple(sizes))


def random_tetromino_partition(
    rng: random.Random,
    board: set[MacroCell],
    piece_count: int,
) -> list[set[MacroCell]] | None:
    if len(board) == piece_count * 4:
        return base.partition_into_tetrominoes(board, piece_count, rng, max_nodes=100_000)
    return partition_macro_board(rng, board, piece_count)


def boundary_edges(regions: list[set[MacroCell]]) -> list[tuple[MacroCell, MacroCell, int, int, str]]:
    owner: dict[MacroCell, int] = {}
    for index, region in enumerate(regions):
        for cell in region:
            owner[cell] = index

    edges: list[tuple[MacroCell, MacroCell, int, int, str]] = []
    for cell, p in owner.items():
        x, y = cell
        for neighbor, direction in (((x + 1, y), "h"), ((x, y + 1), "v")):
            q = owner.get(neighbor)
            if q is not None and q != p:
                edges.append((cell, neighbor, p, q, direction))
    return edges


def make_partition_shapes(
    regions: list[set[MacroCell]],
    rng: random.Random,
    min_area: int,
    max_area: int,
    min_half_cells: int,
    min_horizontal_half_cells: int,
    min_vertical_half_cells: int,
    max_fragile: int,
    transfer_attempts: int,
) -> list[set[Cell]] | None:
    edges = boundary_edges(regions)
    if not edges:
        return None
    valid_pieces: list[set[Cell]] = []
    seen: set[tuple[Cell, ...]] = set()
    for _ in range(transfer_attempts):
        masks_by_piece = [{cell: base.MASK_FULL for cell in region} for region in regions]
        areas = [len(region) * 4 for region in regions]
        split_cells: set[MacroCell] = set()
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)

        # Split many boundary ordinary-cells.  Each split transfers one half-cell
        # from the original owner to the neighboring piece.  Areas may end at
        # 14, 16, or 18 small cells by default.
        split_budget = rng.randint(
            max(3, min_half_cells * len(regions) // 2),
            max(3, len(edges)),
        )

        def try_split(c: MacroCell, d: MacroCell, p: int, q: int, direction: str) -> bool:
            options = [(c, p, q, True), (d, q, p, False)]
            rng.shuffle(options)
            for cell, donor, receiver, positive in options:
                if cell in split_cells:
                    continue
                if areas[donor] - 2 < min_area:
                    continue
                if areas[receiver] + 2 > max_area:
                    continue
                donor_mask, receiver_mask = transfer_masks(direction, positive)
                masks_by_piece[donor][cell] = donor_mask
                masks_by_piece[receiver][cell] = receiver_mask
                areas[donor] -= 2
                areas[receiver] += 2
                split_cells.add(cell)
                return True
            return False

        required_directions = (
            ["v"] * ((min_horizontal_half_cells + 1) // 2)
            + ["h"] * ((min_vertical_half_cells + 1) // 2)
        )
        rng.shuffle(required_directions)
        for required_direction in required_directions:
            for c, d, p, q, direction in shuffled_edges:
                if len(split_cells) >= split_budget:
                    break
                if direction != required_direction:
                    continue
                if try_split(c, d, p, q, direction):
                    break

        for c, d, p, q, direction in shuffled_edges:
            if len(split_cells) >= split_budget:
                break
            try_split(c, d, p, q, direction)

        pieces = [align_cells(base.masks_to_cells(masks)) for masks in masks_by_piece]
        orientation_counts = base.count_half_cell_orientations(pieces)
        if (
            orientation_counts["horizontal_half_cell_count"] < min_horizontal_half_cells
            or orientation_counts["vertical_half_cell_count"] < min_vertical_half_cells
        ):
            continue
        for piece in pieces:
            if not validate_ideal_piece(piece, min_area, max_area, min_half_cells, max_fragile):
                continue
            signature = exact_signature(piece)
            if signature in seen:
                continue
            seen.add(signature)
            valid_pieces.append(piece)
            if len(valid_pieces) >= len(regions):
                return valid_pieces
    return valid_pieces or None


def validate_ideal_piece(
    cells: set[Cell],
    min_area: int,
    max_area: int,
    min_half_cells: int,
    max_fragile: int,
) -> bool:
    if not (min_area <= len(cells) <= max_area):
        return False
    if len(cells) % 2 != 0:
        return False
    if not base.is_connected(cells):
        return False
    if not base.is_legal_half_cell_shape(cells):
        return False
    if base.count_half_cells(cells) < min_half_cells:
        return False
    if base.count_quarter_artifacts(cells) != 0:
        return False
    if base.count_fragile_artifacts(cells) > max_fragile:
        return False
    return True


def build_shape_library(args: argparse.Namespace, macro_board: set[MacroCell]) -> list[ShapeRecord]:
    rng = random.Random(args.seed)
    records: list[ShapeRecord] = []
    exact_seen: set[tuple[Cell, ...]] = set()
    rotational_seen_count: defaultdict[tuple[Cell, ...], int] = defaultdict(int)
    start = time.monotonic()
    attempts = 0

    while len(records) < args.library_target:
        attempts += 1
        if time.monotonic() - start > args.library_time_limit:
            break

        regions = random_tetromino_partition(
            rng, macro_board, args.pieces
        )
        if regions is None:
            continue

        pieces = make_partition_shapes(
            regions,
            rng,
            min_area=args.min_piece_area,
            max_area=args.max_piece_area,
            min_half_cells=args.min_half_cells,
            min_horizontal_half_cells=getattr(args, "min_horizontal_half_cells", 0),
            min_vertical_half_cells=getattr(args, "min_vertical_half_cells", 0),
            max_fragile=args.max_fragile,
            transfer_attempts=args.transfer_attempts,
        )
        if pieces is None:
            continue

        for piece in pieces:
            signature = exact_signature(piece)
            if signature in exact_seen:
                continue
            rot_sig = rotational_signature(piece)
            if rotational_seen_count[rot_sig] >= args.max_per_rotational_family:
                continue
            exact_seen.add(signature)
            rotational_seen_count[rot_sig] += 1
            cells = frozenset(piece)
            records.append(
                ShapeRecord(
                    id=len(records),
                    cells=cells,
                    area=len(cells),
                    half_count=base.count_half_cells(cells),
                    horizontal_half_count=base.count_horizontal_half_cells(cells),
                    vertical_half_count=base.count_vertical_half_cells(cells),
                    fragile_count=base.count_fragile_artifacts(cells),
                    rotational_signature=rot_sig,
                )
            )
            if len(records) >= args.library_target:
                break

        if args.verbose and attempts % 100 == 0:
            print(
                f"library attempts={attempts} shapes={len(records)} "
                f"elapsed={time.monotonic() - start:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    if getattr(args, "allow_identical_pieces", False) and getattr(args, "shape_copies", 1) > 1:
        originals = records[:]
        for _ in range(1, args.shape_copies):
            for record in originals:
                records.append(
                    ShapeRecord(
                        id=len(records),
                        cells=record.cells,
                        area=record.area,
                        half_count=record.half_count,
                        horizontal_half_count=record.horizontal_half_count,
                        vertical_half_count=record.vertical_half_count,
                        fragile_count=record.fragile_count,
                        rotational_signature=record.rotational_signature,
                    )
                )
    return records


def enumerate_shape_placements(
    board: set[Cell],
    shapes: list[ShapeRecord],
) -> list[PlacementRecord]:
    placements: list[PlacementRecord] = []
    board_set = set(board)
    _, _, board_max_x, board_max_y = base.bounds(board_set)
    seen: set[tuple[int, tuple[Cell, ...]]] = set()

    for shape in shapes:
        _, _, shape_max_x, shape_max_y = base.bounds(shape.cells)
        for ty in range(0, board_max_y - shape_max_y + 1, base.SCALE):
            for tx in range(0, board_max_x - shape_max_x + 1, base.SCALE):
                placed = frozenset((x + tx, y + ty) for x, y in shape.cells)
                if placed <= board_set:
                    key = (shape.id, tuple(sorted(placed)))
                    if key in seen:
                        continue
                    seen.add(key)
                    placements.append(
                        PlacementRecord(
                            id=len(placements),
                            shape_id=shape.id,
                            cells=placed,
                        )
                    )
    return placements


def enumerate_phase_preserving_placements(
    board: set[Cell],
    pieces: list[set[Cell]],
) -> tuple[list[list[base.Placement]], dict[Cell, int], int]:
    board_cells = sorted(board)
    index = {cell: i for i, cell in enumerate(board_cells)}
    board_mask = (1 << len(board_cells)) - 1
    _, _, board_max_x, board_max_y = base.bounds(board)

    placements_by_piece: list[list[base.Placement]] = []
    for piece_index, piece in enumerate(pieces):
        aligned = align_cells(piece)
        _, _, piece_max_x, piece_max_y = base.bounds(aligned)
        piece_placements: list[base.Placement] = []
        for ty in range(0, board_max_y - piece_max_y + 1, base.SCALE):
            for tx in range(0, board_max_x - piece_max_x + 1, base.SCALE):
                placed = frozenset((x + tx, y + ty) for x, y in aligned)
                if not placed <= board:
                    continue
                mask = 0
                for cell in placed:
                    mask |= 1 << index[cell]
                piece_placements.append(
                    base.Placement(
                        piece_index=piece_index,
                        cells=placed,
                        mask=mask,
                        origin=(tx, ty),
                        orientation=0,
                    )
                )
        unique: dict[int, base.Placement] = {}
        for placement in piece_placements:
            unique.setdefault(placement.mask, placement)
        placements_by_piece.append(list(unique.values()))
    return placements_by_piece, index, board_mask


def count_solutions_fixed_phase(
    board: set[Cell],
    pieces: list[set[Cell]],
    limit: int,
) -> tuple[int, list[dict[int, frozenset[Cell]]], list[list[base.Placement]]]:
    placements_by_piece, _index, board_mask = enumerate_phase_preserving_placements(board, pieces)
    if any(not placements for placements in placements_by_piece):
        return 0, [], placements_by_piece

    cell_to_placements: dict[int, list[tuple[int, base.Placement]]] = defaultdict(list)
    for piece_index, placements_for_piece in enumerate(placements_by_piece):
        for placement in placements_for_piece:
            m = placement.mask
            while m:
                bit = m & -m
                cell_index = bit.bit_length() - 1
                cell_to_placements[cell_index].append((piece_index, placement))
                m ^= bit

    remaining_start = frozenset(range(len(pieces)))
    solutions: list[dict[int, frozenset[Cell]]] = []
    count = 0

    def search(occupied: int, remaining: frozenset[int], chosen: dict[int, base.Placement]) -> None:
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
        best_options: list[tuple[int, base.Placement]] | None = None
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


def _build_cpsat_model(
    board: set[Cell],
    shapes: list[ShapeRecord],
    placements: list[PlacementRecord],
    args: argparse.Namespace,
) -> tuple[
    cp_model.CpModel,
    dict[int, cp_model.IntVar],
    dict[tuple[int, int], cp_model.IntVar],
    dict[int, PlacementRecord],
] | None:
    if cp_model is None:
        raise SystemExit(
            "OR-Tools is not installed. Run: python -m pip install -r requirements.txt"
        )
    if not placements:
        return None

    model = cp_model.CpModel()
    k_solutions = args.required_solutions
    y = {shape.id: model.NewBoolVar(f"shape_{shape.id}") for shape in shapes}
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for layer in range(k_solutions):
        for placement in placements:
            x[(layer, placement.id)] = model.NewBoolVar(f"x_{layer}_{placement.id}")

    placements_by_shape: defaultdict[int, list[PlacementRecord]] = defaultdict(list)
    placements_by_cell: defaultdict[Cell, list[PlacementRecord]] = defaultdict(list)
    for placement in placements:
        placements_by_shape[placement.shape_id].append(placement)
        for cell in placement.cells:
            placements_by_cell[cell].append(placement)

    model.Add(sum(y.values()) == args.pieces)
    model.Add(sum(shape.area * y[shape.id] for shape in shapes) == len(board))
    model.Add(
        sum(shape.horizontal_half_count * y[shape.id] for shape in shapes)
        >= getattr(args, "min_horizontal_half_cells", 0)
    )
    model.Add(
        sum(shape.vertical_half_count * y[shape.id] for shape in shapes)
        >= getattr(args, "min_vertical_half_cells", 0)
    )

    # Usually we avoid selecting two shapes that are the same physical cut up to
    # rotation.  Ranked search may allow them, but downstream verification counts
    # solutions modulo pure identical-piece swaps.
    if not getattr(args, "allow_identical_pieces", False):
        by_rotational: defaultdict[tuple[Cell, ...], list[int]] = defaultdict(list)
        for shape in shapes:
            by_rotational[shape.rotational_signature].append(shape.id)
        for ids in by_rotational.values():
            if len(ids) > 1:
                model.Add(sum(y[i] for i in ids) <= 1)

    for layer in range(k_solutions):
        for cell in board:
            covering = [x[(layer, p.id)] for p in placements_by_cell[cell]]
            if not covering:
                return None
            model.Add(sum(covering) == 1)

        for shape in shapes:
            model.Add(
                sum(x[(layer, p.id)] for p in placements_by_shape[shape.id])
                == y[shape.id]
            )

    shape_by_id = {shape.id: shape for shape in shapes}
    placements_by_identity: defaultdict[
        tuple[tuple[Cell, ...], tuple[Cell, ...]],
        list[PlacementRecord],
    ] = defaultdict(list)
    for placement in placements:
        shape_signature = shape_by_id[placement.shape_id].rotational_signature
        placement_signature = tuple(sorted(placement.cells))
        placements_by_identity[(shape_signature, placement_signature)].append(placement)

    # Make every requested solution genuinely different from every other one.
    # Difference is measured after collapsing pure swaps of identical physical
    # pieces, matching base.solution_identity_signature() in final verification.
    for a in range(k_solutions):
        for b in range(a + 1, k_solutions):
            different_identity_vars = []
            for identity_index, identity_placements in enumerate(placements_by_identity.values()):
                used_in_a = sum(x[(a, placement.id)] for placement in identity_placements)
                used_in_b = sum(x[(b, placement.id)] for placement in identity_placements)
                is_different = model.NewBoolVar(f"identity_diff_{a}_{b}_{identity_index}")
                model.Add(used_in_a != used_in_b).OnlyEnforceIf(is_different)
                different_identity_vars.append(is_different)
            model.Add(sum(different_identity_vars) >= 1)

    # Prefer more half-cell structure and cleaner pieces.
    model.Maximize(
        sum(shape.half_count * y[shape.id] for shape in shapes)
        + sum(min(shape.horizontal_half_count, shape.vertical_half_count) * 3 * y[shape.id] for shape in shapes)
        - sum(shape.fragile_count * 20 * y[shape.id] for shape in shapes)
    )
    return model, y, x, {placement.id: placement for placement in placements}


def solve_with_cpsat_candidates(
    board: set[Cell],
    shapes: list[ShapeRecord],
    placements: list[PlacementRecord],
    args: argparse.Namespace,
    max_candidates: int = 1,
) -> list[CpsatCandidate]:
    built = _build_cpsat_model(board, shapes, placements, args)
    if built is None:
        return []

    model, y, x, placement_by_id = built
    selected_by_id = {shape.id: shape for shape in shapes}
    start = time.monotonic()
    results: list[CpsatCandidate] = []
    seen: set[tuple[int, ...]] = set()

    for attempt in range(max(1, max_candidates)):
        remaining_time = args.solve_time_limit - (time.monotonic() - start)
        if remaining_time <= 0:
            break

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining_time
        solver.parameters.num_search_workers = args.workers
        solver.parameters.random_seed = (args.seed or 1) + attempt
        solver.parameters.log_search_progress = bool(getattr(args, "solver_log", False) and attempt == 0)

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        selected_ids = tuple(sorted(shape_id for shape_id in y if solver.Value(y[shape_id]) == 1))
        if selected_ids in seen:
            break
        seen.add(selected_ids)
        selected_shapes = tuple(selected_by_id[i] for i in selected_ids)

        proofs: list[CpsatSolutionProof] = []
        for layer in range(args.required_solutions):
            chosen = tuple(
                placement_by_id[placement_id]
                for placement_id in sorted(placement_by_id)
                if solver.Value(x[(layer, placement_id)]) == 1
            )
            proofs.append(CpsatSolutionProof(layer=layer, placements=chosen))
        results.append(CpsatCandidate(shapes=selected_shapes, proofs=tuple(proofs)))

        # Ask CP-SAT for a genuinely different set of physical cuts next time.
        model.Add(sum(y[i] for i in selected_ids) <= args.pieces - 1)

    return results


def solve_with_cpsat(
    board: set[Cell],
    shapes: list[ShapeRecord],
    placements: list[PlacementRecord],
    args: argparse.Namespace,
) -> list[ShapeRecord] | None:
    candidates = solve_with_cpsat_candidates(board, shapes, placements, args, max_candidates=1)
    if not candidates:
        return None

    return list(candidates[0].shapes)


def verify_cpsat_proof_cover(
    board: set[Cell],
    candidate: CpsatCandidate,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if not candidate.proofs:
        return False, "no proof layers"
    selected_shape_ids = {shape.id for shape in candidate.shapes}
    board_set = set(board)

    for proof in candidate.proofs:
        if len(proof.placements) != args.pieces:
            return (
                False,
                f"proof layer {proof.layer} failed: used {len(proof.placements)} placements, expected {args.pieces}",
            )
        used_shape_ids = {placement.shape_id for placement in proof.placements}
        if used_shape_ids != selected_shape_ids:
            return (
                False,
                f"proof layer {proof.layer} failed: used shape ids differ from selected shape ids",
            )

        covered: set[Cell] = set()
        for placement in proof.placements:
            if not placement.cells <= board_set:
                outside = sorted(placement.cells - board_set)[:1]
                return (
                    False,
                    f"proof layer {proof.layer} failed: placement {placement.id} outside board at {outside}",
                )
            overlap = covered & placement.cells
            if overlap:
                return (
                    False,
                    f"proof layer {proof.layer} failed: overlap at cell {sorted(overlap)[0]}",
                )
            covered |= set(placement.cells)
        if covered != board_set:
            return (
                False,
                f"proof layer {proof.layer} failed: covered {len(covered)} cells but board has {len(board_set)}",
            )
    return True, f"{len(candidate.proofs)} proof layer(s) cover the board"


def verify_and_write(
    board: set[Cell],
    selected_shapes: list[ShapeRecord],
    args: argparse.Namespace,
) -> bool:
    pieces = [set(shape.cells) for shape in selected_shapes]
    solution_count, solutions, placements_by_piece = base.count_solutions(
        board,
        pieces,
        allow_rotate=False,
        allow_mirror=False,
        limit=args.solution_count_limit,
    )
    if solution_count < args.required_solutions:
        print(
            f"CP-SAT candidate failed verification: only {solution_count} solutions",
            file=sys.stderr,
        )
        return False
    effective_solutions = []
    seen_solution_signatures: set[object] = set()
    for solution in solutions:
        signature = base.solution_identity_signature(solution, pieces)
        if signature in seen_solution_signatures:
            continue
        seen_solution_signatures.add(signature)
        effective_solutions.append(solution)
    if len(effective_solutions) < args.required_solutions:
        print(
            "CP-SAT candidate failed verification: "
            f"only {len(effective_solutions)} effective solutions after identical-piece swaps",
            file=sys.stderr,
        )
        return False

    rotated_count, _, _ = base.count_solutions(
        board,
        pieces,
        allow_rotate=not args.no_rotate,
        allow_mirror=False,
        limit=args.solution_count_limit,
    )
    analysis = base.analyze_candidate(
        board,
        pieces,
        effective_solutions,
        len(effective_solutions),
        rotated_count,
        placements_by_piece,
        args.required_solutions,
        args.max_solutions,
    )
    candidate = base.Candidate(
        board=board,
        pieces=pieces,
        solutions=effective_solutions,
        solution_count=len(effective_solutions),
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=0,
    )
    base.write_outputs([candidate], args.output_dir)
    print(base.candidate_to_text(candidate, max_solutions=min(4, len(solutions))))
    print(f"Saved ideal candidate to {args.output_dir}")
    return True


def write_library_snapshot(
    shapes: list[ShapeRecord],
    placements: list[PlacementRecord],
    path: Path,
) -> None:
    payload = {
        "shape_count": len(shapes),
        "placement_count": len(placements),
        "shapes": [
            {
                "id": shape.id,
                "area": shape.area,
                "half_count": shape.half_count,
                "horizontal_half_count": shape.horizontal_half_count,
                "vertical_half_count": shape.vertical_half_count,
                "fragile_count": shape.fragile_count,
                "cells": sorted([list(cell) for cell in shape.cells]),
            }
            for shape in shapes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heavy ideal half-polyomino search.")
    parser.add_argument("--pieces", type=int, default=6)
    parser.add_argument("--board-w-macro", type=int, default=6)
    parser.add_argument("--board-h-macro", type=int, default=4)
    parser.add_argument("--board-remove-min", type=int, default=0)
    parser.add_argument("--board-remove-max", type=int, default=0)
    parser.add_argument("--max-board-candidates", type=int, default=20)
    parser.add_argument("--max-board-candidates-per-remove", type=int, default=5000)
    parser.add_argument("--min-boundary-irregularities", type=int, default=2)
    parser.add_argument("--min-boundary-half-notches", type=int, default=2)
    parser.add_argument("--max-boundary-half-notches", type=int, default=4)
    parser.add_argument("--max-board-variants-per-macro", type=int, default=8)
    parser.add_argument("--allow-holes", action="store_true")
    parser.add_argument("--min-piece-area", type=int, default=12)
    parser.add_argument("--max-piece-area", type=int, default=20)
    parser.add_argument("--min-half-cells", type=int, default=0)
    parser.add_argument("--min-horizontal-half-cells", type=int, default=1)
    parser.add_argument("--min-vertical-half-cells", type=int, default=1)
    parser.add_argument("--max-fragile", type=int, default=0)
    parser.add_argument("--required-solutions", type=int, default=4)
    parser.add_argument("--max-solutions", type=int, default=16)
    parser.add_argument("--solution-count-limit", type=int, default=100)
    parser.add_argument("--library-target", type=int, default=2500)
    parser.add_argument("--library-time-limit", type=float, default=900.0)
    parser.add_argument("--solve-time-limit", type=float, default=1800.0)
    parser.add_argument("--transfer-attempts", type=int, default=200)
    parser.add_argument("--max-per-rotational-family", type=int, default=1)
    parser.add_argument("--allow-identical-pieces", action="store_true")
    parser.add_argument("--shape-copies", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("out_ideal"))
    parser.add_argument("--library-json", type=Path, default=Path("out_ideal/library.json"))
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--solver-log", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.pieces <= 0:
        raise SystemExit("--pieces must be positive")
    if args.board_remove_min < 0 or args.board_remove_max < 0:
        raise SystemExit("board remove counts must be non-negative")
    if args.board_remove_min > args.board_remove_max:
        raise SystemExit("--board-remove-min cannot exceed --board-remove-max")
    if args.min_boundary_irregularities < 0 or args.min_boundary_half_notches < 0:
        raise SystemExit("boundary irregularity minimums must be non-negative")
    if args.max_boundary_half_notches < args.min_boundary_half_notches:
        raise SystemExit("--max-boundary-half-notches cannot be smaller than --min-boundary-half-notches")
    if args.max_board_variants_per_macro <= 0:
        raise SystemExit("--max-board-variants-per-macro must be positive")
    min_board_area = (args.board_w_macro * args.board_h_macro - args.board_remove_max) * 4
    max_board_area = (args.board_w_macro * args.board_h_macro - args.board_remove_min) * 4
    if max_board_area < args.pieces * args.min_piece_area:
        raise SystemExit("board area is too small for piece area bounds")
    if min_board_area > args.pieces * args.max_piece_area:
        raise SystemExit("board area is too large for piece area bounds")
    if args.min_piece_area % 2 or args.max_piece_area % 2:
        raise SystemExit("piece area bounds must be even small-cell counts")
    if args.required_solutions < 2:
        raise SystemExit("--required-solutions should be at least 2")
    if args.solution_count_limit < args.required_solutions:
        raise SystemExit("--solution-count-limit must be at least --required-solutions")
    if args.min_horizontal_half_cells < 0 or args.min_vertical_half_cells < 0:
        raise SystemExit("directional half-cell minimums must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if cp_model is None:
        print("OR-Tools is not installed.", file=sys.stderr)
        print("Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.library_json.parent.mkdir(parents=True, exist_ok=True)

    macro_boards = generate_near_rect_macro_boards(args)
    print(f"Generated {len(macro_boards)} near-rect board(s).", file=sys.stderr, flush=True)
    if not macro_boards:
        print("No board candidates match the area and shape constraints.", file=sys.stderr)
        return 1

    for macro_index, macro_board in enumerate(macro_boards, start=1):
        small_boards = generate_boundary_small_boards(args, macro_board)
        if not small_boards:
            continue
        print(
            f"[macro board {macro_index}/{len(macro_boards)}] "
            f"macro_area={len(macro_board)} variants={len(small_boards)} "
            f"score={board_score(macro_board, args.board_w_macro, args.board_h_macro):.1f}",
            file=sys.stderr,
            flush=True,
        )

        print("Building shape library...", file=sys.stderr, flush=True)
        shapes = build_shape_library(args, macro_board)
        print(f"Built {len(shapes)} shapes.", file=sys.stderr, flush=True)
        if len(shapes) < args.pieces:
            print("Not enough shapes for this board.", file=sys.stderr)
            continue

        for variant_index, board in enumerate(small_boards, start=1):
            board_dir = args.output_dir / f"board_{macro_index:03d}_{variant_index:02d}"
            library_json = board_dir / "library.json"
            board_dir.mkdir(parents=True, exist_ok=True)
            metrics = base.board_boundary_metrics(board)
            print(
                f"[board {macro_index}.{variant_index}] "
                f"small_area={len(board)} boundary={metrics['boundary_irregularities']} "
                f"half={metrics['boundary_half_cell_irregularities']}",
                file=sys.stderr,
                flush=True,
            )

            print("Enumerating placements...", file=sys.stderr, flush=True)
            placements = enumerate_shape_placements(board, shapes)
            print(f"Enumerated {len(placements)} placements.", file=sys.stderr, flush=True)
            write_library_snapshot(shapes, placements, library_json)

            print("Solving CP-SAT exact-cover selection...", file=sys.stderr, flush=True)
            selected = solve_with_cpsat(board, shapes, placements, args)
            if selected is None:
                print("No ideal candidate found for this board.", file=sys.stderr)
                continue

            old_output_dir = args.output_dir
            args.output_dir = board_dir
            ok = verify_and_write(board, selected, args)
            args.output_dir = old_output_dir
            if ok:
                print(f"Found candidate in {board_dir}", file=sys.stderr)
                return 0

    print("No ideal candidate found under the current constraints.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
