from types import SimpleNamespace

import generate_half_polyomino as base
import ideal_half_polyomino_search as ideal
import ranked_ideal_search


def _cells(*items):
    return base.masks_to_cells(dict(items))


def _analysis_for(pieces):
    board = set().union(*pieces)
    solution = {index: frozenset(piece) for index, piece in enumerate(pieces)}
    return base.analyze_candidate(
        board=board,
        pieces=pieces,
        solutions=[solution],
        solution_count=1,
        rotated_solution_count=1,
        placements_by_piece=[[] for _ in pieces],
        min_solutions=1,
        max_solutions=1,
    )


def _requirements(
    horizontal=1,
    vertical=1,
    horizontal_contacts=0,
    vertical_contacts=0,
):
    return SimpleNamespace(
        min_horizontal_half_cells=horizontal,
        min_vertical_half_cells=vertical,
        min_horizontal_half_contacts=horizontal_contacts,
        min_vertical_half_contacts=vertical_contacts,
    )


def test_half_cell_orientation_count():
    assert base.is_horizontal_half_mask(base.MASK_TOP)
    assert base.is_horizontal_half_mask(base.MASK_BOTTOM)
    assert not base.is_horizontal_half_mask(base.MASK_FULL)

    assert base.is_vertical_half_mask(base.MASK_LEFT)
    assert base.is_vertical_half_mask(base.MASK_RIGHT)
    assert not base.is_vertical_half_mask(base.MASK_FULL)

    shape = _cells(
        ((0, 0), base.MASK_TOP),
        ((1, 0), base.MASK_BOTTOM),
        ((2, 0), base.MASK_LEFT),
        ((3, 0), base.MASK_RIGHT),
        ((4, 0), base.MASK_FULL),
    )
    assert base.count_horizontal_half_cells(shape) == 2
    assert base.count_vertical_half_cells(shape) == 2


def test_reject_horizontal_only_candidate():
    pieces = [
        _cells(((0, 0), base.MASK_TOP), ((1, 0), base.MASK_BOTTOM)),
    ]
    analysis = _analysis_for(pieces)
    assert analysis.horizontal_half_cell_count == 2
    assert analysis.vertical_half_cell_count == 0
    assert not base.analysis_meets_half_orientation_requirements(
        analysis,
        _requirements(horizontal=1, vertical=1),
    )


def test_reject_vertical_only_candidate():
    pieces = [
        _cells(((0, 0), base.MASK_LEFT), ((1, 0), base.MASK_RIGHT)),
    ]
    analysis = _analysis_for(pieces)
    assert analysis.horizontal_half_cell_count == 0
    assert analysis.vertical_half_cell_count == 2
    assert not base.analysis_meets_half_orientation_requirements(
        analysis,
        _requirements(horizontal=1, vertical=1),
    )


def test_accept_candidate_with_both_orientations():
    pieces = [
        _cells(((0, 0), base.MASK_TOP), ((1, 0), base.MASK_LEFT)),
    ]
    analysis = _analysis_for(pieces)
    assert analysis.horizontal_half_cell_count == 1
    assert analysis.vertical_half_cell_count == 1
    assert base.analysis_meets_half_orientation_requirements(
        analysis,
        _requirements(horizontal=1, vertical=1),
    )


def test_half_cell_contacts_by_orientation():
    pieces = [
        _cells(((0, 0), base.MASK_TOP), ((1, 0), base.MASK_LEFT)),
        _cells(((0, 0), base.MASK_BOTTOM), ((1, 0), base.MASK_RIGHT)),
    ]
    solution = {index: frozenset(piece) for index, piece in enumerate(pieces)}
    contacts = base.count_half_cell_contacts_by_orientation(solution, pieces)
    assert contacts["horizontal"] >= 1
    assert contacts["vertical"] >= 1


def test_json_includes_half_cell_orientation_metrics():
    pieces = [
        _cells(((0, 0), base.MASK_TOP), ((1, 0), base.MASK_LEFT)),
        _cells(((0, 0), base.MASK_BOTTOM), ((1, 0), base.MASK_RIGHT)),
    ]
    board = set().union(*pieces)
    solution = {index: frozenset(piece) for index, piece in enumerate(pieces)}
    analysis = base.analyze_candidate(
        board=board,
        pieces=pieces,
        solutions=[solution],
        solution_count=1,
        rotated_solution_count=1,
        placements_by_piece=[[] for _ in pieces],
        min_solutions=1,
        max_solutions=1,
    )
    candidate = base.Candidate(
        board=board,
        pieces=pieces,
        solutions=[solution],
        solution_count=1,
        placements_by_piece=[[] for _ in pieces],
        score=analysis.difficulty_score,
        analysis=analysis,
        attempts=0,
    )

    payload = base.candidate_to_json(candidate)
    assert payload["piece_areas_small"] == [4, 4]
    assert payload["piece_areas_ordinary_equiv"] == [1.0, 1.0]
    assert payload["pieces"][0]["area_small"] == 4
    assert payload["pieces"][0]["area_ordinary_equiv"] == 1.0
    assert payload["horizontal_half_cell_count"] == 2
    assert payload["vertical_half_cell_count"] == 2
    assert payload["horizontal_half_cell_count_per_piece"] == [1, 1]
    assert payload["vertical_half_cell_count_per_piece"] == [1, 1]
    assert payload["horizontal_half_cell_contacts"] >= 1
    assert payload["vertical_half_cell_contacts"] >= 1


def test_board_boundary_metrics_count_half_and_full_irregularities():
    full_board = base.masks_to_cells(
        {
            (0, 0): base.MASK_FULL,
            (1, 0): base.MASK_FULL,
            (0, 1): base.MASK_FULL,
            (1, 1): base.MASK_FULL,
        }
    )
    assert base.board_boundary_metrics(full_board)["boundary_irregularities"] == 0

    half_notched = base.masks_to_cells(
        {
            (0, 0): base.MASK_RIGHT,
            (1, 0): base.MASK_FULL,
            (0, 1): base.MASK_FULL,
            (1, 1): base.MASK_TOP,
        }
    )
    half_metrics = base.board_boundary_metrics(half_notched)
    assert half_metrics["boundary_irregularities"] == 2
    assert half_metrics["boundary_half_cell_irregularities"] == 2
    assert half_metrics["boundary_full_cell_irregularities"] == 0

    full_cell_notched = base.macro_to_full_small({(0, 0), (1, 0), (0, 1)})
    full_metrics = base.board_boundary_metrics(full_cell_notched)
    assert full_metrics["boundary_irregularities"] == 1
    assert full_metrics["boundary_full_cell_irregularities"] == 1


def test_generate_boundary_small_boards_prefers_half_notches():
    args = SimpleNamespace(
        min_boundary_irregularities=2,
        min_boundary_half_notches=2,
        max_boundary_half_notches=2,
        max_board_variants_per_macro=3,
        max_board_candidates_per_remove=100,
        allow_holes=False,
        pieces=1,
        min_piece_area=12,
        max_piece_area=20,
    )
    boards = ideal.generate_boundary_small_boards(
        args,
        {(0, 0), (1, 0), (0, 1), (1, 1)},
    )
    assert boards
    for board in boards:
        metrics = base.board_boundary_metrics(board)
        assert metrics["boundary_irregularities"] >= 2
        assert metrics["boundary_half_cell_irregularities"] >= 2


def test_cpsat_rejects_pure_identical_piece_swap_proofs():
    if ideal.cp_model is None:
        return

    board = base.macro_to_full_small({(0, 0), (1, 0)})
    piece = base.macro_to_full_small({(0, 0)})
    signature = ideal.rotational_signature(piece)
    shapes = [
        ideal.ShapeRecord(
            id=index,
            cells=frozenset(piece),
            area=len(piece),
            half_count=0,
            horizontal_half_count=0,
            vertical_half_count=0,
            fragile_count=0,
            rotational_signature=signature,
        )
        for index in range(2)
    ]
    placements = ideal.enumerate_shape_placements(board, shapes)
    args = SimpleNamespace(
        required_solutions=2,
        pieces=2,
        min_horizontal_half_cells=0,
        min_vertical_half_cells=0,
        allow_identical_pieces=True,
        solve_time_limit=5.0,
        workers=1,
        seed=1,
        solver_log=False,
    )

    assert ideal.solve_with_cpsat_candidates(board, shapes, placements, args) == []


def test_ranked_piece_area_defaults_are_three_to_five_ordinary_cells():
    args = ranked_ideal_search.parse_args([])
    assert args.min_piece_area == 12
    assert args.max_piece_area == 20
    assert args.min_boundary_irregularities == 2
    assert args.min_boundary_half_notches == 2
    assert args.min_horizontal_half_cells == 1
    assert args.min_vertical_half_cells == 1
    assert args.min_horizontal_half_contacts == 0
    assert args.min_vertical_half_contacts == 0
