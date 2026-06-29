#!/usr/bin/env python3
"""Direct verified search for robust half-cell puzzle candidates.

This script intentionally avoids CP-SAT.  Every generated six-piece set is
checked by the Python exact-cover counter before it is accepted.  It is slower
than ranked_ideal_search.py, but easier to trust and easier to inspect.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import generate_half_polyomino as base
import ideal_half_polyomino_search as ideal
import ranked_ideal_search as ranked


@dataclass
class SearchStats:
    attempts: int = 0
    generated_piece_sets: int = 0
    raw_solution_hits: int = 0
    effective_solution_hits: int = 0
    rejected_fragile: int = 0
    rejected_quarter: int = 0
    rejected_too_few_solutions: int = 0
    accepted: int = 0


def near_rect_boards(
    width: int,
    height: int,
    remove_min: int,
    remove_max: int,
    args: argparse.Namespace,
) -> list[set[tuple[int, int]]]:
    board_args = argparse.Namespace(
        board_w_macro=width,
        board_h_macro=height,
        board_remove_min=remove_min,
        board_remove_max=remove_max,
        max_board_candidates=args.max_boards_per_profile,
        max_board_candidates_per_remove=5000,
        allow_holes=args.allow_holes,
        pieces=args.pieces,
        min_piece_area=args.min_piece_area,
        max_piece_area=args.max_piece_area,
    )
    return ideal.generate_near_rect_macro_boards(board_args)


def build_board_profiles(args: argparse.Namespace) -> list[tuple[str, list[set[tuple[int, int]]]]]:
    profiles = [
        ("6x4_full", [ideal.macro_rectangle(6, 4)]),
        ("5x5_remove1", near_rect_boards(5, 5, 1, 1, args)),
        ("5x5_remove2", near_rect_boards(5, 5, 2, 2, args)),
        ("6x5_remove5-6", near_rect_boards(6, 5, 5, 6, args)),
    ]
    return [(name, boards) for name, boards in profiles if boards]


def compact_random_board(rng: random.Random, args: argparse.Namespace) -> set[tuple[int, int]] | None:
    area = rng.choice((24, 25))
    return base.random_macro_board(
        rng,
        width=args.board_w,
        height=args.board_h,
        area=area,
        allow_holes=args.allow_holes,
        max_tries=200,
    )


def make_covering_piece_set(
    regions: list[set[tuple[int, int]]],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[set[tuple[int, int]]] | None:
    edges = ideal.boundary_edges(regions)
    if not edges:
        return None

    for _ in range(args.transfer_attempts):
        masks_by_piece = [{cell: base.MASK_FULL for cell in region} for region in regions]
        areas = [len(region) * 4 for region in regions]
        split_cells: set[tuple[int, int]] = set()
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        split_budget = rng.randint(
            max(1, args.min_half_cells * len(regions) // 2),
            max(1, len(shuffled_edges)),
        )

        for c, d, p, q, direction in shuffled_edges:
            if len(split_cells) >= split_budget:
                break
            options = [(c, p, q, True), (d, q, p, False)]
            rng.shuffle(options)
            for cell, donor, receiver, positive in options:
                if cell in split_cells:
                    continue
                if areas[donor] - 2 < args.min_piece_area:
                    continue
                if areas[receiver] + 2 > args.max_piece_area:
                    continue
                donor_mask, receiver_mask = ideal.transfer_masks(direction, positive)
                masks_by_piece[donor][cell] = donor_mask
                masks_by_piece[receiver][cell] = receiver_mask
                areas[donor] -= 2
                areas[receiver] += 2
                split_cells.add(cell)
                break

        pieces = [ideal.align_cells(base.masks_to_cells(masks)) for masks in masks_by_piece]
        if all(
            ideal.validate_ideal_piece(
                piece,
                min_area=args.min_piece_area,
                max_area=args.max_piece_area,
                min_half_cells=args.min_half_cells,
                max_fragile=0,
            )
            for piece in pieces
        ):
            return pieces
    return None


def generate_piece_set_for_board(
    rng: random.Random,
    macro_board: set[tuple[int, int]],
    args: argparse.Namespace,
) -> list[set[tuple[int, int]]] | None:
    regions = ideal.random_tetromino_partition(rng, macro_board, args.pieces)
    if regions is None:
        return None
    return make_covering_piece_set(regions, rng, args)


def evaluate_piece_set(
    macro_board: set[tuple[int, int]],
    pieces: list[set[tuple[int, int]]],
    profile_name: str,
    attempt: int,
    args: argparse.Namespace,
) -> tuple[base.Candidate, int, int]:
    board = ideal.macro_to_small_board(macro_board)
    min_x, min_y, _, _ = base.bounds(board)
    board = {(x - min_x, y - min_y) for x, y in board}
    pieces = [ideal.align_cells(piece) for piece in pieces]
    raw_count, raw_solutions, placements_by_piece = ideal.count_solutions_fixed_phase(
        board,
        pieces,
        limit=args.solution_count_limit,
    )
    effective_solutions = ranked.unique_effective_solutions(
        raw_solutions,
        pieces,
        limit=args.solution_count_limit,
    )
    effective_count = len(effective_solutions)
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
        effective_count,
        rotated_count,
        placements_by_piece,
        args.min_solutions,
        args.max_solutions,
    )
    candidate = base.Candidate(
        board=board,
        pieces=pieces,
        solutions=effective_solutions,
        solution_count=effective_count,
        placements_by_piece=placements_by_piece,
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=attempt,
    )
    score, _notes = ranked.rank_candidate(candidate, profile_name)
    candidate.score = score
    return candidate, raw_count, effective_count


def candidate_key(candidate: base.Candidate) -> str:
    return repr(
        (
            sorted(candidate.board),
            [sorted(piece) for piece in candidate.pieces],
        )
    )


def save_candidates_incremental(candidates: list[base.Candidate], output_dir: Path) -> None:
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    base.write_outputs(candidates, output_dir)


def make_near_miss(
    candidate: base.Candidate,
    raw_count: int,
    effective_count: int,
    reject_reason: str,
    profile_name: str,
    args: argparse.Namespace,
) -> ranked.NearMiss:
    return ranked.NearMiss(
        candidate=candidate,
        reject_reason=reject_reason,
        raw_solution_count=raw_count,
        effective_solution_count=effective_count,
        proof_layer_count=0,
        proof_valid=False,
        proof_message="direct search candidate; no CP-SAT proof",
        profile_name=profile_name,
        board_index=0,
        seed=args.seed,
        selected_shape_ids=list(range(len(candidate.pieces))),
        selected_shape_cells=[sorted(piece) for piece in candidate.pieces],
        board_metrics=ranked.small_board_metrics(candidate.board),
        placements_by_piece_counts=[len(placements) for placements in candidate.placements_by_piece],
    )


def log_stats(stats: SearchStats) -> None:
    print(
        " ".join(
            [
                f"attempts={stats.attempts}",
                f"generated_piece_sets={stats.generated_piece_sets}",
                f"raw_solution_hits={stats.raw_solution_hits}",
                f"effective_solution_hits={stats.effective_solution_hits}",
                f"rejected_fragile={stats.rejected_fragile}",
                f"rejected_quarter={stats.rejected_quarter}",
                f"rejected_too_few_solutions={stats.rejected_too_few_solutions}",
                f"accepted={stats.accepted}",
            ]
        ),
        file=sys.stderr,
        flush=True,
    )


def search(args: argparse.Namespace) -> tuple[list[base.Candidate], list[ranked.NearMiss]]:
    rng = random.Random(args.seed)
    start = time.monotonic()
    profiles = build_board_profiles(args)
    candidates: list[base.Candidate] = []
    near_misses: list[ranked.NearMiss] = []
    seen: set[str] = set()
    stats = SearchStats()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    while len(candidates) < args.candidates and time.monotonic() - start < args.time_limit:
        stats.attempts += 1
        profile_name: str
        macro_board: set[tuple[int, int]] | None
        if stats.attempts % 5 == 0:
            profile_name = "compact_random"
            macro_board = compact_random_board(rng, args)
        else:
            profile_name, boards = profiles[(stats.attempts - 1) % len(profiles)]
            macro_board = rng.choice(boards)

        if macro_board is None:
            continue
        pieces = generate_piece_set_for_board(rng, macro_board, args)
        if pieces is None:
            if stats.attempts % 25 == 0:
                log_stats(stats)
            continue

        stats.generated_piece_sets += 1
        candidate, raw_count, effective_count = evaluate_piece_set(
            macro_board,
            pieces,
            profile_name,
            stats.attempts,
            args,
        )

        if candidate.analysis.quarter_artifact_count != 0:
            stats.rejected_quarter += 1
            continue
        if candidate.analysis.fragile_artifact_count != 0:
            stats.rejected_fragile += 1
            continue
        if raw_count > 0:
            stats.raw_solution_hits += 1
        if effective_count > 0:
            stats.effective_solution_hits += 1

        reject_reason = ""
        if effective_count < args.min_solutions:
            stats.rejected_too_few_solutions += 1
            reject_reason = f"effective_solution_count={effective_count}<min={args.min_solutions}"
        elif not args.allow_identical_pieces and candidate.analysis.duplicate_piece_count != 0:
            reject_reason = f"duplicate_piece_count={candidate.analysis.duplicate_piece_count}"
        elif effective_count > args.max_solutions:
            reject_reason = f"effective_solution_count={effective_count}>max={args.max_solutions}"

        key = candidate_key(candidate)
        if reject_reason:
            miss = make_near_miss(candidate, raw_count, effective_count, reject_reason, profile_name, args)
            ranked.add_near_miss(near_misses, miss, args)
        elif key not in seen:
            seen.add(key)
            candidates.append(candidate)
            stats.accepted += 1
            save_candidates_incremental(candidates, args.output_dir)
            if args.verbose:
                print(
                    f"accepted #{stats.accepted} attempt={stats.attempts} "
                    f"profile={profile_name} effective={effective_count} score={candidate.score:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

        if args.write_near_misses and near_misses and stats.attempts % args.near_miss_save_interval == 0:
            ranked.write_near_miss_outputs(near_misses, args)
        if stats.attempts % 25 == 0:
            log_stats(stats)

    if candidates:
        save_candidates_incremental(candidates, args.output_dir)
    ranked.write_near_miss_outputs(near_misses, args)
    return candidates, near_misses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct verified half-cell search.")
    parser.add_argument("--pieces", type=int, default=6)
    parser.add_argument("--min-half-cells", type=int, default=1)
    parser.add_argument("--min-solutions", type=int, default=2)
    parser.add_argument("--max-solutions", type=int, default=100)
    parser.add_argument("--allow-identical-pieces", action="store_true", default=True)
    parser.add_argument("--no-identical-pieces", dest="allow_identical_pieces", action="store_false")
    parser.add_argument("--board-w", type=int, default=6)
    parser.add_argument("--board-h", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output-dir", type=Path, default=Path("out_direct"))
    parser.add_argument("--min-piece-area", type=int, default=14)
    parser.add_argument("--max-piece-area", type=int, default=18)
    parser.add_argument("--solution-count-limit", type=int, default=100)
    parser.add_argument("--transfer-attempts", type=int, default=120)
    parser.add_argument("--max-boards-per-profile", type=int, default=24)
    parser.add_argument("--allow-holes", action="store_true")
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--write-near-misses", action="store_true", default=True)
    parser.add_argument("--near-miss-limit", type=int, default=20)
    parser.add_argument("--accept-one-solution-nearmiss", action="store_true", default=True)
    parser.add_argument("--near-miss-output-dir", type=Path, default=Path("out_direct/near_miss"))
    parser.add_argument("--near-miss-save-interval", type=int, default=25)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.pieces != 6:
        raise SystemExit("verified direct search currently expects --pieces 6")
    if args.solution_count_limit < args.min_solutions:
        raise SystemExit("--solution-count-limit must be at least --min-solutions")
    if args.min_piece_area % 2 or args.max_piece_area % 2:
        raise SystemExit("piece area bounds must be even small-cell counts")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    candidates, near_misses = search(args)
    if candidates:
        print(f"Saved {len(candidates)} accepted candidate(s) to {args.output_dir}")
        return 0
    if near_misses:
        print(f"No accepted candidates. Saved near misses to {args.near_miss_output_dir}", file=sys.stderr)
        return 1
    print("No accepted candidates or near misses found.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
