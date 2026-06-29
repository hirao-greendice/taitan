#!/usr/bin/env python3
"""One-shot ranked search for robust half-cell puzzle candidates.

This runner keeps the non-negotiable constraints hard, then ranks the rest.

Hard filters:
  - legal half-cell masks only; no quarter/three-quarter/diagonal/L masks
  - paper fragility count must be 0
  - effective fixed-orientation solution count must be high enough

Important scoring rule:
  Identical physical pieces are allowed, but pure swaps of identical pieces are
  collapsed before counting solutions. Duplicate pieces are a penalty, not a
  hard rejection unless --no-identical-pieces is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import generate_half_polyomino as base
import ideal_half_polyomino_search as ideal


BOARD_ROUGHNESS_CHOICES = ("neat", "balanced", "rough", "wild")
BOARD_RANK_WEIGHTS = {
    "neat": (620.0, 42.0, 1.45, 220.0, 10.0),
    "balanced": (540.0, 38.0, 1.55, 190.0, 8.0),
    "rough": (260.0, 14.0, 2.05, 60.0, 5.0),
    "wild": (120.0, 6.0, 2.6, 25.0, 3.0),
}


@dataclass
class RankedResult:
    candidate: base.Candidate
    board_index: int
    profile_name: str
    rank_score: float
    notes: list[str]


def macro_bounds(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_x = min(x for x, _ in cells)
    max_x = max(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_y = max(y for _, y in cells)
    return min_x, min_y, max_x, max_y


def small_board_metrics(board: set[tuple[int, int]]) -> dict[str, float]:
    macro = {(x // base.SCALE, y // base.SCALE) for x, y in board}
    min_x, min_y, max_x, max_y = macro_bounds(macro)
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    fill = len(macro) / (bbox_w * bbox_h)
    perimeter = ideal.macro_perimeter(macro)
    ideal_perimeter = 2 * (bbox_w + bbox_h)
    aspect = max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h))
    narrow_corridor = ideal.narrow_corridor_penalty(macro)
    return {
        "macro_area": len(macro),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "fill": fill,
        "perimeter_extra": max(0, perimeter - ideal_perimeter),
        "aspect": aspect,
        "narrow_corridor": narrow_corridor,
    }


def unique_effective_solutions(
    solutions: list[dict[int, frozenset[tuple[int, int]]]],
    pieces: list[set[tuple[int, int]]],
    limit: int,
) -> list[dict[int, frozenset[tuple[int, int]]]]:
    seen: set[object] = set()
    unique: list[dict[int, frozenset[tuple[int, int]]]] = []
    for solution in solutions:
        signature = base.solution_identity_signature(solution, pieces)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(solution)
        if len(unique) >= limit:
            break
    return unique


def rank_candidate(
    candidate: base.Candidate,
    profile_name: str,
    board_roughness: str = "balanced",
    profile_board_size: tuple[int, int] | None = None,
) -> tuple[float, list[str]]:
    analysis = candidate.analysis
    metrics = small_board_metrics(candidate.board)
    pieces = candidate.pieces
    areas = [len(piece) for piece in pieces]
    half_counts = [base.count_half_cells(piece) for piece in pieces]

    score = 0.0
    notes: list[str] = []

    if analysis.fragile_artifact_count != 0:
        score -= 100_000
        notes.append("fragile>0")
    if analysis.quarter_artifact_count != 0:
        score -= 100_000
        notes.append("quarter>0")

    if analysis.solution_count >= 4:
        score += 280
        notes.append(f"effective {analysis.solution_count} fixed solutions")
    elif analysis.solution_count == 3:
        score += 150
        notes.append("effective 3 fixed solutions")
    elif analysis.solution_count == 2:
        score += 75
        notes.append("effective 2 fixed solutions")
    else:
        score -= 500
        notes.append("too few fixed solutions")

    fill_weight, perimeter_weight, aspect_limit, aspect_weight, corridor_weight = BOARD_RANK_WEIGHTS.get(
        board_roughness,
        BOARD_RANK_WEIGHTS["balanced"],
    )
    score += metrics["fill"] * fill_weight
    score -= metrics["perimeter_extra"] * perimeter_weight
    score -= max(0.0, metrics["aspect"] - aspect_limit) * aspect_weight
    score -= metrics["narrow_corridor"] * corridor_weight
    score -= abs(metrics["macro_area"] - 24) * 8

    total_half = sum(half_counts)
    score += min(total_half, 26) * 12
    score += min(half_counts) * 22

    score -= (max(areas) - min(areas)) * 8
    score -= sum(abs(area - 16) for area in areas) * 2

    if analysis.duplicate_piece_count:
        score -= analysis.duplicate_piece_count * 55
        notes.append(f"duplicate pieces {analysis.duplicate_piece_count}")

    avg_candidates = analysis.average_piece_candidates
    if avg_candidates < 2:
        score -= 100
        notes.append("few placement choices")
    elif avg_candidates > 45:
        score -= 35
        notes.append("many placement choices")
    else:
        score += min(avg_candidates, 18) * 4

    if profile_board_size is None:
        note_w = int(metrics["bbox_w"])
        note_h = int(metrics["bbox_h"])
        remove_count = int(metrics["bbox_w"] * metrics["bbox_h"] - metrics["macro_area"])
    else:
        note_w, note_h = profile_board_size
        remove_count = int(note_w * note_h - metrics["macro_area"])
    notes.append(
        f"board {note_w}x{note_h} remove={remove_count} roughness={board_roughness} "
        f"perimeter_extra={int(metrics['perimeter_extra'])} fill={metrics['fill']:.2f} "
        f"narrow_corridor={int(metrics['narrow_corridor'])}"
    )
    notes.append(f"half cells total {total_half}")
    notes.append(profile_name)
    return score, notes


def make_candidate(
    board: set[tuple[int, int]],
    selected_shapes: list[ideal.ShapeRecord],
    args: argparse.Namespace,
) -> base.Candidate | None:
    pieces = [set(shape.cells) for shape in selected_shapes]
    raw_solution_count, raw_solutions, placements_by_piece = base.count_solutions(
        board,
        pieces,
        allow_rotate=False,
        allow_mirror=False,
        limit=args.solution_count_limit,
    )
    if raw_solution_count == 0:
        return None

    effective_solutions = unique_effective_solutions(
        raw_solutions,
        pieces,
        limit=args.solution_count_limit,
    )
    effective_solution_count = len(effective_solutions)
    if effective_solution_count < args.min_acceptable_solutions:
        return None

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
        effective_solution_count,
        rotated_count,
        placements_by_piece,
        args.min_acceptable_solutions,
        args.target_solutions,
    )
    if analysis.quarter_artifact_count != 0:
        return None
    if analysis.fragile_artifact_count != 0:
        return None
    if analysis.duplicate_piece_count != 0 and not args.allow_identical_pieces:
        return None

    return base.Candidate(
        board=board,
        pieces=pieces,
        solutions=effective_solutions,
        solution_count=effective_solution_count,
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=0,
    )


def trim_solver_input(
    shapes: list[ideal.ShapeRecord],
    placements: list[ideal.PlacementRecord],
    args: argparse.Namespace,
) -> tuple[list[ideal.ShapeRecord], list[ideal.PlacementRecord]]:
    if not shapes or not placements:
        return shapes, placements
    if len(shapes) <= args.max_solver_shapes and len(placements) <= args.max_solver_placements:
        return shapes, placements

    placements_by_shape: defaultdict[int, int] = defaultdict(int)
    for placement in placements:
        placements_by_shape[placement.shape_id] += 1

    def shape_key(shape: ideal.ShapeRecord) -> tuple[float, int, int, int]:
        placement_count = placements_by_shape[shape.id]
        if placement_count == 0:
            return (-1_000_000, 0, 0, 0)
        flexible_bonus = min(placement_count, 28)
        overflex_penalty = max(0, placement_count - 45)
        return (
            shape.half_count * 100 + flexible_bonus - overflex_penalty,
            -abs(shape.area - 16),
            -shape.fragile_count,
            -shape.id,
        )

    kept: list[ideal.ShapeRecord] = []
    kept_ids: set[int] = set()
    placement_budget = 0
    for shape in sorted(shapes, key=shape_key, reverse=True):
        count = placements_by_shape[shape.id]
        if count == 0:
            continue
        if len(kept) >= args.max_solver_shapes:
            break
        if placement_budget + count > args.max_solver_placements and len(kept) >= args.pieces:
            continue
        kept.append(shape)
        kept_ids.add(shape.id)
        placement_budget += count

    kept_placements = [placement for placement in placements if placement.shape_id in kept_ids]
    return kept, kept_placements


def profile_args(base_args: argparse.Namespace, profile: dict[str, object]) -> argparse.Namespace:
    args = copy.copy(base_args)
    for key, value in profile.items():
        setattr(args, key, value)
    return args


def board_limited_args(
    args: argparse.Namespace,
    base_args: argparse.Namespace,
    started_at: float,
) -> argparse.Namespace | None:
    elapsed = time.monotonic() - started_at
    remaining = base_args.total_time_limit - elapsed
    if remaining <= 5:
        return None

    board_args = copy.copy(args)
    requested = args.library_time_limit + args.solve_time_limit
    board_budget = min(requested, remaining)
    if board_budget < 2:
        return None

    library_budget = min(args.library_time_limit, max(1.0, board_budget * 0.45))
    solve_budget = min(args.solve_time_limit, max(1.0, board_budget - library_budget))
    board_args.library_time_limit = library_budget
    board_args.solve_time_limit = solve_budget
    return board_args


def search_profile(
    base_args: argparse.Namespace,
    profile: dict[str, object],
    started_at: float,
    results: list[RankedResult],
) -> None:
    args = profile_args(base_args, profile)
    profile_name = str(profile["name"])
    macro_boards = ideal.generate_near_rect_macro_boards(args)
    if args.verbose:
        print(f"[{profile_name}] boards={len(macro_boards)}", file=sys.stderr, flush=True)

    for board_index, macro_board in enumerate(macro_boards, start=1):
        if time.monotonic() - started_at > base_args.total_time_limit:
            return
        if len(results) >= base_args.keep_candidates and not base_args.keep_searching:
            return

        board_args = board_limited_args(args, base_args, started_at)
        if board_args is None:
            return

        board = ideal.macro_to_small_board(macro_board)
        if args.verbose:
            print(
                f"[{profile_name}] board {board_index}/{len(macro_boards)} "
                f"macro_area={len(macro_board)} "
                f"lib={board_args.library_time_limit:.0f}s solve={board_args.solve_time_limit:.0f}s",
                file=sys.stderr,
                flush=True,
            )

        shapes = ideal.build_shape_library(board_args, macro_board)
        if args.verbose:
            print(f"[{profile_name}] built shapes={len(shapes)}", file=sys.stderr, flush=True)
        if len(shapes) < board_args.pieces:
            continue

        placements = ideal.enumerate_shape_placements(board, shapes)
        if args.verbose:
            print(f"[{profile_name}] enumerated placements={len(placements)}", file=sys.stderr, flush=True)
        if not placements:
            continue

        before_shapes = len(shapes)
        before_placements = len(placements)
        shapes, placements = trim_solver_input(shapes, placements, board_args)
        if len(shapes) < board_args.pieces or not placements:
            continue

        if args.verbose:
            print(
                f"[{profile_name}] solver input shapes={len(shapes)}/{before_shapes} "
                f"placements={len(placements)}/{before_placements}",
                file=sys.stderr,
                flush=True,
            )

        selected_sets = ideal.solve_with_cpsat_candidates(
            board,
            shapes,
            placements,
            board_args,
            max_candidates=board_args.solver_candidates_per_board,
        )
        if not selected_sets and board_args.single_cover_fallback:
            one_cover_args = copy.copy(board_args)
            one_cover_args.required_solutions = 1
            one_cover_args.solve_time_limit = min(
                board_args.solve_time_limit,
                board_args.single_cover_solve_time_limit,
            )
            selected_sets = ideal.solve_with_cpsat_candidates(
                board,
                shapes,
                placements,
                one_cover_args,
                max_candidates=board_args.solver_candidates_per_board,
            )
        if args.verbose:
            print(f"[{profile_name}] solver returned {len(selected_sets)} set(s)", file=sys.stderr, flush=True)
        for selected in selected_sets:
            selected_shapes = list(selected.shapes) if hasattr(selected, "shapes") else selected
            candidate = make_candidate(board, selected_shapes, board_args)
            if candidate is None:
                continue

            rank_score, notes = rank_candidate(
                candidate,
                profile_name,
                getattr(board_args, "board_roughness", "balanced"),
                (int(board_args.board_w_macro), int(board_args.board_h_macro)),
            )
            results.append(
                RankedResult(
                    candidate=candidate,
                    board_index=board_index,
                    profile_name=profile_name,
                    rank_score=rank_score,
                    notes=notes,
                )
            )
            results.sort(key=lambda result: result.rank_score, reverse=True)
            if args.verbose:
                print(
                    f"accepted score={rank_score:.1f} "
                    f"solutions={candidate.solution_count} notes={' / '.join(notes)}",
                    file=sys.stderr,
                    flush=True,
                )
            if len(results) >= base_args.keep_candidates and not base_args.keep_searching:
                return


def effort_caps(effort: str) -> list[int]:
    if effort == "fast":
        return [1, 3, 5, 5, 1, 4, 4]
    if effort == "relaxed":
        # Candidate-first mode.  Try fewer boards per profile, but make the
        # profiles much easier: sol2 first, half1 first, clean boards first.
        return [1, 2, 4, 4, 4, 1, 4, 4]
    if effort == "deep":
        return [1, 20, 30, 30, 1, 30, 30]
    if effort == "extreme":
        return [1, 60, 90, 90, 1, 80, 80]
    return [1, 8, 12, 12, 1, 12, 12]


def rough_effort_caps(effort: str) -> list[int]:
    if effort == "relaxed":
        return [8, 8, 8, 10, 10, 10]
    if effort == "fast":
        return [3, 3, 3, 3, 3, 3]
    if effort == "deep":
        return [30, 30, 30, 36, 36, 36]
    if effort == "extreme":
        return [90, 90, 90, 120, 120, 120]
    return [12, 12, 12, 16, 16, 16]


def rough_profile_roughness(args: argparse.Namespace) -> str:
    return "wild" if args.board_roughness == "wild" else "rough"


def neat_profile_roughness(args: argparse.Namespace) -> str:
    return "neat" if args.board_roughness == "neat" else "balanced"


def build_profiles(args: argparse.Namespace) -> list[dict[str, object]]:
    caps = effort_caps(args.effort)
    rough_caps = rough_effort_caps(args.effort)
    if args.effort == "relaxed":
        # These profiles match the real puzzle requirements:
        # - at least 2 effective fixed-orientation solutions
        # - legal half-cell masks only
        # - paper fragility remains hard max_fragile=0 below
        # - board neatness is handled by rank_candidate(), not by rejecting early
        neat_specs = [
            ("6x4_full_half1_sol2", 6, 4, 0, 0, caps[0], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_full_half1_sol2", 5, 5, 0, 0, caps[1], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_remove0-2_half1_sol2", 5, 5, 0, 2, caps[2], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_remove0-3_half1_sol2", 5, 5, 0, 3, caps[3], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("6x5_remove5-6_half1_sol2", 6, 5, 5, 6, caps[4], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("6x4_full_half2_sol2", 6, 4, 0, 0, caps[5], 2, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_remove0-3_half2_sol2", 5, 5, 0, 3, caps[6], 2, 2, args.shape_copies, neat_profile_roughness(args)),
            ("6x5_remove5-6_half2_sol2", 6, 5, 5, 6, caps[7], 2, 2, args.shape_copies, neat_profile_roughness(args)),
        ]
    else:
        neat_specs = [
            ("5x5_full_half3_sol4", 5, 5, 0, 0, caps[0], 3, 4, 1, neat_profile_roughness(args)),
            ("5x5_remove0-2_half3_sol4", 5, 5, 0, 2, caps[1], 3, 4, 1, neat_profile_roughness(args)),
            ("5x5_remove0-3_half3_sol3", 5, 5, 0, 3, caps[2], 3, 3, 1, neat_profile_roughness(args)),
            ("5x5_remove0-3_half2_sol2", 5, 5, 0, 3, caps[3], 2, 2, args.shape_copies, neat_profile_roughness(args)),
            ("6x4_full_half2_sol4", 6, 4, 0, 0, caps[4], 2, 4, 1, neat_profile_roughness(args)),
            ("5x5_remove0-3_half2_sol4", 5, 5, 0, 3, caps[5], 2, 4, 1, neat_profile_roughness(args)),
            ("6x5_remove5-6_half2_sol4", 6, 5, 5, 6, caps[6], 2, 4, 1, neat_profile_roughness(args)),
        ]

    rough_specs = [
        ("5x6_remove5-8_half1_sol2", 5, 6, 5, 8, rough_caps[0], 1, 2, args.shape_copies, rough_profile_roughness(args)),
        ("6x5_remove6-8_half1_sol2", 6, 5, 6, 8, rough_caps[1], 1, 2, args.shape_copies, rough_profile_roughness(args)),
        ("7x4_remove4-6_half1_sol2", 7, 4, 4, 6, rough_caps[2], 1, 2, args.shape_copies, rough_profile_roughness(args)),
        ("7x5_remove10-12_half1_sol2", 7, 5, 10, 12, rough_caps[3], 1, 2, args.shape_copies, rough_profile_roughness(args)),
        ("6x6_remove10-12_half1_sol2", 6, 6, 10, 12, rough_caps[4], 1, 2, args.shape_copies, rough_profile_roughness(args)),
        ("8x4_remove7-9_half1_sol2", 8, 4, 7, 9, rough_caps[5], 1, 2, args.shape_copies, rough_profile_roughness(args)),
    ]

    if args.board_roughness == "neat":
        specs = neat_specs
    elif args.board_roughness in ("rough", "wild") and args.effort == "relaxed":
        fallback_name = "5x5_remove0-3_half1_sol2"
        specs = rough_specs
        specs += [spec for spec in neat_specs if spec[0] == fallback_name]
        specs += [spec for spec in neat_specs if spec[0] != fallback_name]
    elif args.board_roughness in ("rough", "wild"):
        specs = rough_specs + neat_specs
    else:
        specs = neat_specs + rough_specs

    profiles: list[dict[str, object]] = []
    for (
        name,
        board_w,
        board_h,
        remove_min,
        remove_max,
        max_boards,
        min_half,
        required_solutions,
        shape_copies,
        board_roughness,
    ) in specs:
        profiles.append(
            {
                "name": name,
                "board_w_macro": board_w,
                "board_h_macro": board_h,
                "board_remove_min": remove_min,
                "board_remove_max": remove_max,
                "board_roughness": board_roughness,
                "max_board_candidates": max_boards,
                "min_half_cells": min_half,
                "required_solutions": required_solutions,
                "library_target": args.library_target,
                "max_fragile": 0,
                "allow_identical_pieces": args.allow_identical_pieces,
                "shape_copies": shape_copies if args.allow_identical_pieces else 1,
            }
        )
    return profiles


def write_ranked_outputs(results: list[RankedResult], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [result.candidate for result in results[:limit]]
    for result in results[:limit]:
        result.candidate.score = result.rank_score
    base.write_outputs(candidates, output_dir)

    summary = []
    for i, result in enumerate(results[:limit], start=1):
        metrics = small_board_metrics(result.candidate.board)
        summary.append(
            {
                "rank": i,
                "score": result.rank_score,
                "profile": result.profile_name,
                "board_index": result.board_index,
                "board_macro_area": metrics["macro_area"],
                "board_bbox": [metrics["bbox_w"], metrics["bbox_h"]],
                "board_fill": metrics["fill"],
                "board_extra_perimeter": metrics["perimeter_extra"],
                "solution_count_effective_fixed": result.candidate.solution_count,
                "rotated_solution_count_raw": result.candidate.analysis.rotated_solution_count,
                "half_cell_count_per_piece": result.candidate.analysis.half_cell_count_per_piece,
                "total_half_cell_count": result.candidate.analysis.total_half_cell_count,
                "quarter_artifact_count": result.candidate.analysis.quarter_artifact_count,
                "fragile_artifact_count": result.candidate.analysis.fragile_artifact_count,
                "duplicate_piece_count": result.candidate.analysis.duplicate_piece_count,
                "notes": result.notes,
            }
        )
    (output_dir / "ranking.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_random_fallback_results(
    args: argparse.Namespace,
    started_at: float,
    results: list[RankedResult],
) -> None:
    if args.no_random_fallback:
        return
    if len(results) >= args.keep_candidates and not args.keep_searching:
        return

    remaining = args.total_time_limit - (time.monotonic() - started_at)
    fallback_time = min(args.random_fallback_time_limit, remaining)
    if fallback_time <= 10:
        return

    wanted = max(1, args.keep_candidates - len(results))
    fallback_args = argparse.Namespace(
        pieces=args.pieces,
        board_w=6,
        board_h=5,
        min_solutions=args.min_acceptable_solutions,
        max_solutions=args.solution_count_limit,
        allow_mirror=False,
        no_rotate=args.no_rotate,
        seed=args.seed + 10_000,
        candidates=wanted,
        time_limit=fallback_time,
        output_dir=args.output_dir,
        allow_holes=args.allow_holes,
        allow_identical_pieces=args.allow_identical_pieces,
        min_half_cells=args.fallback_min_half_cells,
        solution_count_limit=args.solution_count_limit,
        verbose=args.verbose,
    )
    if args.verbose:
        print(
            f"[random_fallback] time={fallback_time:.0f}s candidates={wanted}",
            file=sys.stderr,
            flush=True,
        )

    for candidate in base.generate_candidates(fallback_args):
        effective_solutions = unique_effective_solutions(
            candidate.solutions,
            candidate.pieces,
            limit=args.solution_count_limit,
        )
        if len(effective_solutions) < args.min_acceptable_solutions:
            continue
        rotated_count, _, _ = base.count_solutions(
            candidate.board,
            candidate.pieces,
            allow_rotate=not args.no_rotate,
            allow_mirror=False,
            limit=args.solution_count_limit,
        )
        analysis = base.analyze_candidate(
            candidate.board,
            candidate.pieces,
            effective_solutions,
            len(effective_solutions),
            rotated_count,
            candidate.placements_by_piece,
            args.min_acceptable_solutions,
            args.target_solutions,
        )
        if analysis.quarter_artifact_count != 0:
            continue
        if analysis.fragile_artifact_count != 0:
            continue
        if analysis.duplicate_piece_count != 0 and not args.allow_identical_pieces:
            continue

        candidate.solutions = effective_solutions
        candidate.solution_count = len(effective_solutions)
        candidate.analysis = analysis
        rank_score, notes = rank_candidate(candidate, "random_fallback", args.board_roughness)
        results.append(
            RankedResult(
                candidate=candidate,
                board_index=0,
                profile_name="random_fallback",
                rank_score=rank_score,
                notes=notes,
            )
        )
        results.sort(key=lambda result: result.rank_score, reverse=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot ranked ideal search.")
    parser.add_argument("--pieces", type=int, default=6)
    parser.add_argument("--min-piece-area", type=int, default=14)
    parser.add_argument("--max-piece-area", type=int, default=18)
    parser.add_argument("--target-solutions", type=int, default=4)
    parser.add_argument("--min-acceptable-solutions", type=int, default=2)
    parser.add_argument("--solution-count-limit", type=int, default=200)
    parser.add_argument("--library-target", type=int, default=2500)
    parser.add_argument("--library-time-limit", type=float, default=180.0)
    parser.add_argument("--solve-time-limit", type=float, default=180.0)
    parser.add_argument("--total-time-limit", type=float, default=3600.0)
    parser.add_argument("--transfer-attempts", type=int, default=120)
    parser.add_argument("--max-per-rotational-family", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--effort", choices=("fast", "balanced", "relaxed", "deep", "extreme"), default="balanced")
    parser.add_argument("--board-roughness", choices=BOARD_ROUGHNESS_CHOICES, default="balanced")
    parser.add_argument("--output-dir", type=Path, default=Path("out_ranked"))
    parser.add_argument("--keep-candidates", type=int, default=12)
    parser.add_argument("--keep-searching", action="store_true")
    parser.add_argument("--allow-identical-pieces", action="store_true", default=True)
    parser.add_argument("--no-identical-pieces", dest="allow_identical_pieces", action="store_false")
    parser.add_argument("--shape-copies", type=int, default=2)
    parser.add_argument("--solver-candidates-per-board", type=int, default=3)
    parser.add_argument("--single-cover-fallback", action="store_true", default=True)
    parser.add_argument("--no-single-cover-fallback", dest="single_cover_fallback", action="store_false")
    parser.add_argument("--single-cover-solve-time-limit", type=float, default=45.0)
    parser.add_argument("--max-solver-shapes", type=int, default=2500)
    parser.add_argument("--max-solver-placements", type=int, default=22_000)
    parser.add_argument("--max-board-candidates-per-remove", type=int, default=2000)
    parser.add_argument("--random-fallback-time-limit", type=float, default=600.0)
    parser.add_argument("--fallback-min-half-cells", type=int, default=1)
    parser.add_argument("--no-random-fallback", action="store_true")
    parser.add_argument("--allow-holes", action="store_true")
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--solver-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if ideal.cp_model is None:
        print("OR-Tools is not installed. Run: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    started_at = time.monotonic()
    results: list[RankedResult] = []
    for profile in build_profiles(args):
        if time.monotonic() - started_at > args.total_time_limit:
            break
        search_profile(args, profile, started_at, results)
        if len(results) >= args.keep_candidates and not args.keep_searching:
            break

    add_random_fallback_results(args, started_at, results)
    write_ranked_outputs(results, args.output_dir, args.keep_candidates)
    if not results:
        print("No ranked candidates found. The output gallery is empty.", file=sys.stderr)
        return 1
    print(f"Saved {min(len(results), args.keep_candidates)} ranked candidate(s) to {args.output_dir}")
    print(f"Best score: {results[0].rank_score:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
