# Colab / long-running ranked search

This search is intentionally heavy.  It ranks candidates automatically instead
of asking you to manually change rules after every failed run.

## What is searched

Hard constraints:

- 6 pieces
- Paper fragility must be 0
- No 1/4, 3/4, diagonal-half, or L masks
- Identical physical cuts are allowed
- Pure swaps of identical pieces do not count as distinct solutions
- Piece areas may vary from 14 to 18 small cells
  - 14 = 3.5 ordinary cells
  - 16 = 4 ordinary cells
  - 18 = 4.5 ordinary cells
- Fixed-orientation solutions

Ranking preferences:

- At least 4 fixed-orientation solutions, but 2 or 3 can still be ranked lower
- Board close to a rectangle, with low perimeter noise
- More legal half-cell masks
- Balanced piece areas
- Fewer identical pieces

## Run locally

```bash
python -m pip install -r requirements.txt
```

Debug-first PowerShell run:

```powershell
python ranked_ideal_search.py --effort relaxed --total-time-limit 900 --library-target 800 --library-time-limit 40 --solve-time-limit 45 --solver-candidates-per-board 12 --max-solver-shapes 3000 --max-solver-placements 50000 --random-fallback-time-limit 120 --fallback-min-half-cells 1 --workers 8 --keep-candidates 12 --seed 303 --output-dir out_debug --write-near-misses --near-miss-limit 20 --accept-one-solution-nearmiss --verbose
```

Open:

- `out_debug/index.html` for accepted candidates
- `out_debug/near_miss/index.html` for rejected-but-interesting candidates

Direct verified PowerShell run:

```powershell
python verified_direct_search.py --min-half-cells 1 --min-solutions 2 --max-solutions 100 --allow-identical-pieces --board-w 6 --board-h 5 --time-limit 1800 --candidates 20 --seed 1001 --output-dir out_direct --verbose
```

Deep local run for a gaming PC:

```powershell
python ranked_ideal_search.py --effort deep --library-target 6000 --library-time-limit 600 --solve-time-limit 600 --single-cover-solve-time-limit 120 --total-time-limit 14400 --solver-candidates-per-board 6 --max-solver-shapes 4500 --max-solver-placements 45000 --keep-candidates 20 --keep-searching --workers 8 --output-dir out_ranked_deep --verbose
```

## Run in Google Colab

Create a new Colab notebook and run these cells.

### 1. Upload or clone the project

If the repo is already on GitHub:

```python
!git clone YOUR_REPO_URL taitan
%cd taitan
```

If you have not pushed it, upload these files to Colab first:

- `generate_half_polyomino.py`
- `ideal_half_polyomino_search.py`
- `ranked_ideal_search.py`
- `requirements.txt`

Then:

```python
%cd /content
```

### 2. Install dependencies

```python
!pip install -r requirements.txt
```

### 3. Run a serious search

```python
!python ranked_ideal_search.py \
  --effort balanced \
  --total-time-limit 7200 \
  --workers 2 \
  --keep-candidates 12 \
  --output-dir out_ranked \
  --verbose
```

Debug-first Colab run with near misses:

```python
!python ranked_ideal_search.py \
  --effort relaxed \
  --total-time-limit 900 \
  --library-target 800 \
  --library-time-limit 40 \
  --solve-time-limit 45 \
  --solver-candidates-per-board 12 \
  --max-solver-shapes 3000 \
  --max-solver-placements 50000 \
  --random-fallback-time-limit 120 \
  --fallback-min-half-cells 1 \
  --workers 2 \
  --keep-candidates 12 \
  --seed 303 \
  --output-dir out_debug \
  --write-near-misses \
  --near-miss-limit 20 \
  --accept-one-solution-nearmiss \
  --verbose
```

Direct verified Colab run:

```python
!python verified_direct_search.py \
  --min-half-cells 1 \
  --min-solutions 2 \
  --max-solutions 100 \
  --allow-identical-pieces \
  --board-w 6 \
  --board-h 5 \
  --time-limit 1800 \
  --candidates 20 \
  --seed 1001 \
  --output-dir out_direct \
  --verbose
```

Longer Colab run:

```python
!python ranked_ideal_search.py \
  --effort deep \
  --library-target 5000 \
  --library-time-limit 600 \
  --solve-time-limit 600 \
  --single-cover-solve-time-limit 120 \
  --total-time-limit 14400 \
  --solver-candidates-per-board 6 \
  --max-solver-shapes 4000 \
  --max-solver-placements 40000 \
  --keep-candidates 20 \
  --keep-searching \
  --workers 2 \
  --output-dir out_ranked_deep \
  --verbose
```

### 4. Download results

```python
!zip -r out_ranked.zip out_ranked
from google.colab import files
files.download("out_ranked.zip")
```

Open `out_ranked/index.html` after downloading.

Use `--solver-log` only when you want the very noisy OR-Tools internal log.

## Relaxation knobs

The ranked search already tries these automatically where appropriate:

- min half-cell count 3 and 2
- required solutions 4, 3, and 2
- 5x5 near-rectangle boards
- 6x5 near-rectangle boards

It never relaxes paper fragility.
