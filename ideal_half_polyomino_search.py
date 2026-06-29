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
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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
    fragile_count: int
    rotational_signature: tuple[Cell, ...]


@dataclass(frozen=True)
class PlacementRecord:
    id: int
    shape_id: int
    cells: frozenset[Cell]


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


def random_tetromino_partition(
    rng: random.Random,
    width_macro: int,
    height_macro: int,
    piece_count: int,
) -> list[set[MacroCell]] | None:
    board = macro_rectangle(width_macro, height_macro)
    return base.partition_into_tetrominoes(board, piece_count, rng, max_nodes=100_000)


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
    max_fragile: int,
    transfer_attempts: int,
) -> list[set[Cell]] | None:
    for _ in range(transfer_attempts):
        masks_by_piece = [{cell: base.MASK_FULL for cell in region} for region in regions]
        areas = [len(region) * 4 for region in regions]
        split_cells: set[MacroCell] = set()
        edges = boundary_edges(regions)
        rng.shuffle(edges)

        # Split many boundary ordinary-cells.  Each split transfers one half-cell
        # from the original owner to the neighboring piece.  Areas may end at
        # 14, 16, or 18 small cells by default.
        split_budget = rng.randint(
            max(3, min_half_cells * len(regions) // 2),
            max(3, len(edges)),
        )
        for c, d, p, q, direction in edges:
            if len(split_cells) >= split_budget:
                break
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
                break

        pieces = [align_cells(base.masks_to_cells(masks)) for masks in masks_by_piece]
        if all(
            validate_ideal_piece(piece, min_area, max_area, min_half_cells, max_fragile)
            for piece in pieces
        ):
            return pieces
    return None


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


def build_shape_library(args: argparse.Namespace) -> list[ShapeRecord]:
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
            rng, args.board_w_macro, args.board_h_macro, args.pieces
        )
        if regions is None:
            continue

        pieces = make_partition_shapes(
            regions,
            rng,
            min_area=args.min_piece_area,
            max_area=args.max_piece_area,
            min_half_cells=args.min_half_cells,
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


def solve_with_cpsat(
    board: set[Cell],
    shapes: list[ShapeRecord],
    placements: list[PlacementRecord],
    args: argparse.Namespace,
) -> list[ShapeRecord] | None:
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

    # Do not select two shapes that are the same physical cut up to rotation.
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

    # Make every requested solution genuinely different from every other one.
    for a in range(k_solutions):
        for b in range(a + 1, k_solutions):
            same_vars = []
            for placement in placements:
                z = model.NewBoolVar(f"same_{a}_{b}_{placement.id}")
                model.Add(z <= x[(a, placement.id)])
                model.Add(z <= x[(b, placement.id)])
                model.Add(z >= x[(a, placement.id)] + x[(b, placement.id)] - 1)
                same_vars.append(z)
            model.Add(sum(same_vars) <= args.pieces - 1)

    # Prefer more half-cell structure and cleaner pieces.
    model.Maximize(
        sum(shape.half_count * y[shape.id] for shape in shapes)
        - sum(shape.fragile_count * 20 * y[shape.id] for shape in shapes)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.solve_time_limit
    solver.parameters.num_search_workers = args.workers
    solver.parameters.random_seed = args.seed or 1
    solver.parameters.log_search_progress = args.verbose

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    selected_ids = [shape.id for shape in shapes if solver.Value(y[shape.id]) == 1]
    selected_by_id = {shape.id: shape for shape in shapes}
    return [selected_by_id[i] for i in selected_ids]


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
        solutions,
        solution_count,
        rotated_count,
        placements_by_piece,
        args.required_solutions,
        args.max_solutions,
    )
    candidate = base.Candidate(
        board=board,
        pieces=pieces,
        solutions=solutions,
        solution_count=solution_count,
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
    parser.add_argument("--min-piece-area", type=int, default=14)
    parser.add_argument("--max-piece-area", type=int, default=18)
    parser.add_argument("--min-half-cells", type=int, default=3)
    parser.add_argument("--max-fragile", type=int, default=0)
    parser.add_argument("--required-solutions", type=int, default=4)
    parser.add_argument("--max-solutions", type=int, default=16)
    parser.add_argument("--solution-count-limit", type=int, default=100)
    parser.add_argument("--library-target", type=int, default=2500)
    parser.add_argument("--library-time-limit", type=float, default=900.0)
    parser.add_argument("--solve-time-limit", type=float, default=1800.0)
    parser.add_argument("--transfer-attempts", type=int, default=200)
    parser.add_argument("--max-per-rotational-family", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("out_ideal"))
    parser.add_argument("--library-json", type=Path, default=Path("out_ideal/library.json"))
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.pieces <= 0:
        raise SystemExit("--pieces must be positive")
    board_area = args.board_w_macro * args.board_h_macro * 4
    if not (args.pieces * args.min_piece_area <= board_area <= args.pieces * args.max_piece_area):
        raise SystemExit("board area is incompatible with piece area bounds")
    if args.min_piece_area % 2 or args.max_piece_area % 2:
        raise SystemExit("piece area bounds must be even small-cell counts")
    if args.required_solutions < 2:
        raise SystemExit("--required-solutions should be at least 2")
    if args.solution_count_limit < args.required_solutions:
        raise SystemExit("--solution-count-limit must be at least --required-solutions")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if cp_model is None:
        print("OR-Tools is not installed.", file=sys.stderr)
        print("Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    board = rectangle_board(args.board_w_macro * 2, args.board_h_macro * 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.library_json.parent.mkdir(parents=True, exist_ok=True)

    print("Building shape library...", file=sys.stderr, flush=True)
    shapes = build_shape_library(args)
    print(f"Built {len(shapes)} shapes.", file=sys.stderr, flush=True)
    if len(shapes) < args.pieces:
        print("Not enough shapes; relax constraints or increase time.", file=sys.stderr)
        return 1

    print("Enumerating placements...", file=sys.stderr, flush=True)
    placements = enumerate_shape_placements(board, shapes)
    print(f"Enumerated {len(placements)} placements.", file=sys.stderr, flush=True)
    write_library_snapshot(shapes, placements, args.library_json)

    print("Solving CP-SAT exact-cover selection...", file=sys.stderr, flush=True)
    selected = solve_with_cpsat(board, shapes, placements, args)
    if selected is None:
        print("No ideal candidate found under the current constraints.", file=sys.stderr)
        return 1

    ok = verify_and_write(board, selected, args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
