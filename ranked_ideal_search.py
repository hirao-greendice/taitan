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
from html import escape
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


@dataclass
class NearMiss:
    candidate: base.Candidate
    reject_reason: str
    raw_solution_count: int
    effective_solution_count: int
    proof_layer_count: int
    proof_valid: bool
    proof_message: str
    profile_name: str
    board_index: int
    seed: int
    selected_shape_ids: list[int]
    selected_shape_cells: list[list[tuple[int, int]]]
    board_metrics: dict[str, float]
    placements_by_piece_counts: list[int]


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
    score += min(analysis.horizontal_half_cell_count, analysis.vertical_half_cell_count) * 50
    score -= abs(analysis.horizontal_half_cell_count - analysis.vertical_half_cell_count) * 20
    if analysis.horizontal_half_cell_count == 0:
        score -= 500
    if analysis.vertical_half_cell_count == 0:
        score -= 500
    score += analysis.horizontal_half_cell_contacts * 40
    score += analysis.vertical_half_cell_contacts * 40
    if analysis.horizontal_half_cell_contacts == 0:
        score -= 250
    if analysis.vertical_half_cell_contacts == 0:
        score -= 250

    average_area = sum(areas) / max(1, len(areas))
    score -= sum(abs(area - average_area) for area in areas) * 0.75

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
    notes.append(
        f"half cells total {total_half} "
        f"h={analysis.horizontal_half_cell_count} v={analysis.vertical_half_cell_count} "
        f"contacts_h={analysis.horizontal_half_cell_contacts} contacts_v={analysis.vertical_half_cell_contacts}"
    )
    notes.append(profile_name)
    return score, notes


def make_candidate(
    board: set[tuple[int, int]],
    selected: ideal.CpsatCandidate,
    args: argparse.Namespace,
) -> tuple[base.Candidate | None, NearMiss | None, str]:
    pieces = [set(shape.cells) for shape in selected.shapes]
    proof_valid, proof_message = ideal.verify_cpsat_proof_cover(board, selected, args)
    raw_solution_count, raw_solutions, placements_by_piece = ideal.count_solutions_fixed_phase(
        board,
        pieces,
        limit=args.solution_count_limit,
    )

    effective_solutions = unique_effective_solutions(
        raw_solutions,
        pieces,
        limit=args.solution_count_limit,
    )
    effective_solution_count = len(effective_solutions)

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

    candidate = base.Candidate(
        board=board,
        pieces=pieces,
        solutions=effective_solutions,
        solution_count=effective_solution_count,
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=0,
    )

    reject_reasons: list[str] = []
    if not proof_valid:
        reject_reasons.append(f"proof_invalid:{proof_message}")
    if raw_solution_count == 0:
        reject_reasons.append("raw_solution_count=0")
    if effective_solution_count < args.min_acceptable_solutions:
        reject_reasons.append(
            f"effective_solution_count={effective_solution_count}<min={args.min_acceptable_solutions}"
        )
    if analysis.quarter_artifact_count != 0:
        reject_reasons.append(f"quarter_artifact_count={analysis.quarter_artifact_count}")
    if analysis.fragile_artifact_count != 0:
        reject_reasons.append(f"fragile_artifact_count={analysis.fragile_artifact_count}")
    if analysis.duplicate_piece_count != 0 and not args.allow_identical_pieces:
        reject_reasons.append(f"duplicate_piece_count={analysis.duplicate_piece_count}")
    piece_areas = [len(piece) for piece in pieces]
    if any(area < args.min_piece_area or area > args.max_piece_area for area in piece_areas):
        reject_reasons.append(
            "piece_area_out_of_bounds "
            f"areas={piece_areas} min={args.min_piece_area} max={args.max_piece_area}"
        )
    min_horizontal = getattr(args, "min_horizontal_half_cells", 0)
    min_vertical = getattr(args, "min_vertical_half_cells", 0)
    min_horizontal_contacts = getattr(args, "min_horizontal_half_contacts", 0)
    min_vertical_contacts = getattr(args, "min_vertical_half_contacts", 0)
    if analysis.horizontal_half_cell_count < min_horizontal:
        reject_reasons.append(
            "not_enough_horizontal_half_cells "
            f"horizontal={analysis.horizontal_half_cell_count} min={min_horizontal}"
        )
    if analysis.vertical_half_cell_count < min_vertical:
        reject_reasons.append(
            "not_enough_vertical_half_cells "
            f"vertical={analysis.vertical_half_cell_count} min={min_vertical}"
        )
    if analysis.horizontal_half_cell_contacts < min_horizontal_contacts:
        reject_reasons.append(
            "not_enough_horizontal_half_contacts "
            f"horizontal_contacts={analysis.horizontal_half_cell_contacts} min={min_horizontal_contacts}"
        )
    if analysis.vertical_half_cell_contacts < min_vertical_contacts:
        reject_reasons.append(
            "not_enough_vertical_half_contacts "
            f"vertical_contacts={analysis.vertical_half_cell_contacts} min={min_vertical_contacts}"
        )

    near_miss = NearMiss(
        candidate=candidate,
        reject_reason="; ".join(reject_reasons) if reject_reasons else "",
        raw_solution_count=raw_solution_count,
        effective_solution_count=effective_solution_count,
        proof_layer_count=len(selected.proofs),
        proof_valid=proof_valid,
        proof_message=proof_message,
        profile_name=getattr(args, "profile_name", ""),
        board_index=getattr(args, "board_index", 0),
        seed=args.seed,
        selected_shape_ids=[shape.id for shape in selected.shapes],
        selected_shape_cells=[sorted(shape.cells) for shape in selected.shapes],
        board_metrics=small_board_metrics(board),
        placements_by_piece_counts=[len(placements) for placements in placements_by_piece],
    )

    if reject_reasons:
        return None, near_miss, near_miss.reject_reason

    return candidate, None, "accepted"


def should_store_near_miss(miss: NearMiss, args: argparse.Namespace) -> bool:
    if not args.write_near_misses:
        return False
    min_solutions = getattr(args, "min_acceptable_solutions", getattr(args, "min_solutions", 2))
    if miss.proof_valid and miss.effective_solution_count < min_solutions:
        return True
    if miss.raw_solution_count >= 1 and miss.effective_solution_count < min_solutions:
        return True
    if args.accept_one_solution_nearmiss and miss.effective_solution_count >= 1:
        return True
    if miss.candidate.analysis.fragile_artifact_count == 1:
        return True
    if "duplicate_piece_count" in miss.reject_reason:
        return True
    if "not_enough_horizontal_half" in miss.reject_reason:
        return True
    if "not_enough_vertical_half" in miss.reject_reason:
        return True
    if miss.board_metrics["fill"] >= 0.88 and miss.effective_solution_count < min_solutions:
        return True
    return False


def add_near_miss(near_misses: list[NearMiss], miss: NearMiss | None, args: argparse.Namespace) -> None:
    if miss is None or not should_store_near_miss(miss, args):
        return
    near_misses.append(miss)
    near_misses.sort(
        key=lambda item: (
            item.effective_solution_count,
            item.raw_solution_count,
            item.board_metrics["fill"],
            -item.board_metrics["perimeter_extra"],
        ),
        reverse=True,
    )
    del near_misses[args.near_miss_limit :]


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
    near_misses: list[NearMiss],
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
        board_args.profile_name = profile_name
        board_args.board_index = board_index

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
            candidate, near_miss, reject_reason = make_candidate(board, selected, board_args)
            if candidate is None:
                add_near_miss(near_misses, near_miss, base_args)
                if args.verbose:
                    print(
                        f"[{profile_name}] rejected: {reject_reason}",
                        file=sys.stderr,
                        flush=True,
                    )
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
    if effort == "relaxed":
        return [1, 8, 12, 16, 16, 10, 14]
    if effort == "fast":
        return [1, 3, 5, 5, 1, 4, 4]
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
        neat_specs = [
            ("6x4_full_half1_sol2", 6, 4, 0, 0, caps[0], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_full_half1_sol2", 5, 5, 0, 0, caps[1], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_remove0-2_half1_sol2", 5, 5, 0, 2, caps[2], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_remove0-3_half1_sol2", 5, 5, 0, 3, caps[3], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("6x5_remove5-6_half1_sol2", 6, 5, 5, 6, caps[4], 1, 2, args.shape_copies, neat_profile_roughness(args)),
            ("6x4_full_half2_sol2", 6, 4, 0, 0, caps[5], 2, 2, args.shape_copies, neat_profile_roughness(args)),
            ("5x5_remove0-3_half2_sol2", 5, 5, 0, 3, caps[6], 2, 2, args.shape_copies, neat_profile_roughness(args)),
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
                "piece_areas_small": [len(piece) for piece in result.candidate.pieces],
                "piece_areas_ordinary_equiv": [len(piece) / 4.0 for piece in result.candidate.pieces],
                "solution_count_effective_fixed": result.candidate.solution_count,
                "rotated_solution_count_raw": result.candidate.analysis.rotated_solution_count,
                "half_cell_count_per_piece": result.candidate.analysis.half_cell_count_per_piece,
                "total_half_cell_count": result.candidate.analysis.total_half_cell_count,
                "horizontal_half_cell_count": result.candidate.analysis.horizontal_half_cell_count,
                "vertical_half_cell_count": result.candidate.analysis.vertical_half_cell_count,
                "horizontal_half_cell_count_per_piece": result.candidate.analysis.horizontal_half_cell_count_per_piece,
                "vertical_half_cell_count_per_piece": result.candidate.analysis.vertical_half_cell_count_per_piece,
                "horizontal_half_cell_contacts": result.candidate.analysis.horizontal_half_cell_contacts,
                "vertical_half_cell_contacts": result.candidate.analysis.vertical_half_cell_contacts,
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


def near_miss_to_json(miss: NearMiss, rank: int) -> dict[str, object]:
    analysis = miss.candidate.analysis
    return {
        "rank": rank,
        "reject_reason": miss.reject_reason,
        "raw_solution_count": miss.raw_solution_count,
        "effective_solution_count": miss.effective_solution_count,
        "proof_layer_count": miss.proof_layer_count,
        "proof_valid": miss.proof_valid,
        "proof_message": miss.proof_message,
        "quarter_artifact_count": analysis.quarter_artifact_count,
        "fragile_artifact_count": analysis.fragile_artifact_count,
        "duplicate_piece_count": analysis.duplicate_piece_count,
        "horizontal_half_cell_count": analysis.horizontal_half_cell_count,
        "vertical_half_cell_count": analysis.vertical_half_cell_count,
        "horizontal_half_cell_count_per_piece": analysis.horizontal_half_cell_count_per_piece,
        "vertical_half_cell_count_per_piece": analysis.vertical_half_cell_count_per_piece,
        "horizontal_half_cell_contacts": analysis.horizontal_half_cell_contacts,
        "vertical_half_cell_contacts": analysis.vertical_half_cell_contacts,
        "piece_areas_small": [len(piece) for piece in miss.candidate.pieces],
        "piece_areas_ordinary_equiv": [len(piece) / 4.0 for piece in miss.candidate.pieces],
        "board_metrics": miss.board_metrics,
        "profile_name": miss.profile_name,
        "board_index": miss.board_index,
        "seed": miss.seed,
        "selected_shape_ids": miss.selected_shape_ids,
        "selected_shape_cells": [
            [list(cell) for cell in cells]
            for cells in miss.selected_shape_cells
        ],
        "placements_by_piece_counts": miss.placements_by_piece_counts,
        "half_cell_count_per_piece": analysis.half_cell_count_per_piece,
        "total_half_cell_count": analysis.total_half_cell_count,
    }


def write_near_miss_html(near_misses: list[NearMiss], path: Path) -> None:
    cards = []
    for index, miss in enumerate(near_misses, start=1):
        candidate = miss.candidate
        metrics = miss.board_metrics
        solution_svg = (
            base.svg_solution(candidate.board, candidate.solutions[0])
            if candidate.solutions
            else "<div class='empty'>No phase-preserving fixed solution counted.</div>"
        )
        cards.append(
            f"""
            <article class="card">
              <div class="badge">NEAR MISS / NOT FINAL</div>
              <h2>Near Miss {index}</h2>
              <p class="reason">{escape(miss.reject_reason)}</p>
              <div class="metrics">
                <span>raw {miss.raw_solution_count}</span>
                <span>effective {miss.effective_solution_count}</span>
                <span>proof layers {miss.proof_layer_count}</span>
                <span>proof {str(miss.proof_valid).lower()}</span>
                <span>fragile {candidate.analysis.fragile_artifact_count}</span>
                <span>quarter {candidate.analysis.quarter_artifact_count}</span>
                <span>duplicate {candidate.analysis.duplicate_piece_count}</span>
                <span>h half {candidate.analysis.horizontal_half_cell_count}</span>
                <span>v half {candidate.analysis.vertical_half_cell_count}</span>
                <span>h per piece {escape(','.join(str(v) for v in candidate.analysis.horizontal_half_cell_count_per_piece))}</span>
                <span>v per piece {escape(','.join(str(v) for v in candidate.analysis.vertical_half_cell_count_per_piece))}</span>
                <span>h contact {candidate.analysis.horizontal_half_cell_contacts}</span>
                <span>v contact {candidate.analysis.vertical_half_cell_contacts}</span>
                <span>areas small {escape(','.join(str(len(piece)) for piece in candidate.pieces))}</span>
                <span>areas ordinary {escape(','.join(f'{len(piece) / 4.0:.1f}' for piece in candidate.pieces))}</span>
                <span>fill {metrics['fill']:.2f}</span>
                <span>perimeter+ {metrics['perimeter_extra']:.0f}</span>
              </div>
              <div class="grid">
                <section><h3>Board</h3>{base.svg_board(candidate.board)}</section>
                <section><h3>First Counted Solution</h3>{solution_svg}</section>
                <section class="pieces"><h3>Pieces</h3>{base.svg_pieces(candidate.pieces)}</section>
              </div>
              <pre>{escape(miss.proof_message)}</pre>
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Near Miss Gallery</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f4ee; color: #222; }}
    header {{ padding: 24px 28px; background: #24211d; color: #fff; }}
    header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    header p {{ margin: 0; color: #ddd; }}
    main {{ padding: 20px; display: grid; gap: 18px; }}
    .card {{ border: 1px solid #d2cab8; background: #fff; border-radius: 8px; padding: 18px; }}
    .badge {{ display: inline-block; background: #8f1d1d; color: #fff; font-weight: 700; padding: 6px 9px; border-radius: 4px; }}
    h2 {{ margin: 12px 0 6px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .reason {{ font-family: Consolas, monospace; background: #f4eee5; padding: 8px; border-radius: 4px; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }}
    .metrics span {{ border: 1px solid #d7d0c2; border-radius: 4px; padding: 5px 8px; background: #fbfaf7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; align-items: start; }}
    svg {{ max-width: 100%; height: auto; border: 1px solid #ddd4c4; }}
    .pieces {{ grid-column: 1 / -1; }}
    .empty {{ min-height: 120px; display: grid; place-items: center; border: 1px dashed #cbbf9f; color: #6f624d; }}
    pre {{ white-space: pre-wrap; background: #28241f; color: #f2eadc; padding: 10px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Near Miss Gallery</h1>
    <p>NEAR MISS / NOT FINAL. These candidates failed at least one acceptance rule.</p>
  </header>
  <main>
    {''.join(cards) if cards else '<p>No near misses were captured.</p>'}
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_near_miss_outputs(near_misses: list[NearMiss], args: argparse.Namespace) -> None:
    if not args.write_near_misses:
        return
    output_dir = args.near_miss_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    limited = near_misses[: args.near_miss_limit]
    payload = [near_miss_to_json(miss, index) for index, miss in enumerate(limited, start=1)]
    (output_dir / "near_miss.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_near_miss_html(limited, output_dir / "index.html")


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
        min_horizontal_half_cells=args.min_horizontal_half_cells,
        min_vertical_half_cells=args.min_vertical_half_cells,
        min_horizontal_half_contacts=args.min_horizontal_half_contacts,
        min_vertical_half_contacts=args.min_vertical_half_contacts,
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
        if any(len(piece) < args.min_piece_area or len(piece) > args.max_piece_area for piece in candidate.pieces):
            continue
        if analysis.horizontal_half_cell_count < args.min_horizontal_half_cells:
            continue
        if analysis.vertical_half_cell_count < args.min_vertical_half_cells:
            continue
        if analysis.horizontal_half_cell_contacts < args.min_horizontal_half_contacts:
            continue
        if analysis.vertical_half_cell_contacts < args.min_vertical_half_contacts:
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
    parser.add_argument("--min-piece-area", type=int, default=12)
    parser.add_argument("--max-piece-area", type=int, default=20)
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
    parser.add_argument("--effort", choices=("relaxed", "fast", "balanced", "deep", "extreme"), default="relaxed")
    parser.add_argument("--board-roughness", choices=BOARD_ROUGHNESS_CHOICES, default="balanced")
    parser.add_argument("--min-horizontal-half-cells", type=int, default=2)
    parser.add_argument("--min-vertical-half-cells", type=int, default=2)
    parser.add_argument("--min-horizontal-half-contacts", type=int, default=1)
    parser.add_argument("--min-vertical-half-contacts", type=int, default=1)
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
    parser.add_argument("--write-near-misses", action="store_true")
    parser.add_argument("--near-miss-limit", type=int, default=20)
    parser.add_argument("--accept-one-solution-nearmiss", action="store_true")
    parser.add_argument("--near-miss-output-dir", type=Path, default=Path("out_debug/near_miss"))
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
    near_misses: list[NearMiss] = []
    for profile in build_profiles(args):
        if time.monotonic() - started_at > args.total_time_limit:
            break
        search_profile(args, profile, started_at, results, near_misses)
        if len(results) >= args.keep_candidates and not args.keep_searching:
            break

    add_random_fallback_results(args, started_at, results)
    write_ranked_outputs(results, args.output_dir, args.keep_candidates)
    write_near_miss_outputs(near_misses, args)
    if not results:
        if near_misses and args.write_near_misses:
            print(f"No ranked candidates found. Saved near misses to {args.near_miss_output_dir}", file=sys.stderr)
        else:
            print("No ranked candidates found. The output gallery is empty.", file=sys.stderr)
        return 1
    print(f"Saved {min(len(results), args.keep_candidates)} ranked candidate(s) to {args.output_dir}")
    print(f"Best score: {results[0].rank_score:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
