# Colab / long-running ideal search

This search is intentionally heavy.  It does **not** use identical-piece swaps
unless you explicitly use the older generator with `--allow-identical-pieces`.

## What is searched

Default ideal constraints:

- 6 pieces
- Full rectangular board: ordinary `6 x 4` cells = small `12 x 8` cells
- Piece areas may vary from 14 to 18 small cells
  - 14 = 3.5 ordinary cells
  - 16 = 4 ordinary cells
  - 18 = 4.5 ordinary cells
- Fixed-orientation solutions
- At least 4 distinct complete tilings
- No identical physical cuts, including rotations
- Legal half-cell masks only
- Minimum half-cell masks per piece: default 3
- Paper fragility count: default 0

## Run locally

```bash
python -m pip install -r requirements.txt
python ideal_half_polyomino_search.py ^
  --library-target 5000 ^
  --library-time-limit 1800 ^
  --solve-time-limit 3600 ^
  --workers 8 ^
  --output-dir out_ideal ^
  --verbose
```

PowerShell one-line version:

```powershell
python -m pip install -r requirements.txt
python ideal_half_polyomino_search.py --library-target 5000 --library-time-limit 1800 --solve-time-limit 3600 --workers 8 --output-dir out_ideal --verbose
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
!python ideal_half_polyomino_search.py \
  --library-target 12000 \
  --library-time-limit 3600 \
  --solve-time-limit 7200 \
  --workers 8 \
  --output-dir out_ideal \
  --verbose
```

### 4. Download results

```python
!zip -r out_ideal.zip out_ideal
from google.colab import files
files.download("out_ideal.zip")
```

Open `out_ideal/index.html` after downloading.

## Relaxation knobs

Use these only if the strict search finds no candidate after a long run.

```bash
--min-half-cells 2
--max-fragile 1
--board-w-macro 7 --board-h-macro 4
--required-solutions 3
```

Recommended relaxation order:

1. Increase `--library-target`, `--library-time-limit`, and `--solve-time-limit`.
2. Try `--min-half-cells 2`.
3. Try `--max-fragile 1`.
4. Try a slightly larger board such as `7 x 4` ordinary cells.

Do **not** use identical pieces for final puzzle candidates.
