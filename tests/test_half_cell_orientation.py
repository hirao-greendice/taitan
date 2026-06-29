from types import SimpleNamespace

import generate_half_polyomino as base
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


def test_ranked_piece_area_defaults_are_three_to_five_ordinary_cells():
    args = ranked_ideal_search.parse_args([])
    assert args.min_piece_area == 12
    assert args.max_piece_area == 20
