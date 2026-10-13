# alma_pipeline_smart_mitigation

Score-based "smart" size mitigation for the ALMA imaging pipeline.

## Overview

The ALMA pipeline stage `hif_checkproductsize` makes sure that the imaging products of a MOUS stay within three size limits. If the predicted sizes are too large, it *mitigates* them. The original implementation (`CheckProductSizeHeuristics.mitigate_sizes()` in `checkproductsize.py`) does this as a fixed cascade: it tries the mitigation steps one after the other (channel binning, field of view, cell size, then dropping fields, then dropping spectral windows), recomputes the sizes after every step and stops as soon as the limits are met.

This project replaces the cascade with a **score-based planner**:

* The effect of every mitigation lever is expressed as the fraction of the size that is left (the *size fraction*).
* Every lever gets a *score* from its size fraction and a *weight* that encodes how acceptable the lever is (moderate levers such as channel binning are cheap, removing spws or fields is expensive).
* **All** possible combinations of mitigations are enumerated, with their resulting sizes and a final score between 0 and 1. The plan with the highest score that meets all limits is the result.
* The full, sorted list of plans can be returned for diagnostics.

The priorities of the levers are therefore a handful of numbers (`LEVER_WEIGHTS`) instead of the order of hard-coded steps in the code.

## How it works

### Limits

| Limit | Meaning |
|-------|---------|
| `maxcubesize` | Per-spw cube target (soft). Levers are applied to a spw's cube until it is below this value, if possible. Never removes a spw by itself. |
| `maxproductsize` | Budget for the summed product size of all kept spws and fields (hard). |
| `maxcubelimit` | Per-spw hard limit on the cube size. A non-representative spw above it cannot be kept; if the representative spw cannot get below it, mitigation fails. At most one non-representative spw above half of this limit is kept. |

`-1` means "no limit", as in the pipeline inputs.

### Mitigation levers

| Lever | Acts on | What it does | Size fraction left |
|-------|---------|--------------|--------------------|
| `n_bin` | each cube | Average channels by 2 (only for eligible `nchan`) | `1/nbin` |
| `PB` | each cube | Tighten the field of view down to a floor of PB = 0.7 (single pointings only, not mosaics) | `(0.7/PB)^2` |
| `cell` | each cube | Enlarge the cell size down to a floor of 3.25 ppb (robust = 2.0) or 3.0 ppb | `(floor/ppb)^2` |
| `ROI` | each cube | Image only part of the spectral window (optional, not yet in the pipeline; off unless `use_roi=True`) | `n_roi/n_total` |
| spws | product total | Remove spectral windows (the representative spw is always kept) | share of the product size in the kept spws |
| fields | product total | Remove fields (the representative field is always kept) | `n_kept/n_total` (equal-sized fields) |

All size fractions are between 0 and 1, where **1 means nothing is removed**. The cube levers (`n_bin`, `PB`, `cell`, `ROI`) shrink both the cubes and the products. Removing spws or fields only shrinks the product total, never an individual cube. Levers are assumed to compound independently (their size fractions multiply).

### Scores

The score of one lever is

```
score = size_fraction ** (1 / weight)
```

* With weight 1 this is the **base score**, the size fraction itself.
* The weight modifies the base score multiplicatively (in the exponent). A larger weight softens the penalty, a smaller one sharpens it. A lever that removes nothing always scores 1.
* The weights are set in `LEVER_WEIGHTS` and can be overridden per call: `plan(..., weights={"spws": 5})`. Defaults: `n_bin`, `PB`, `cell` = 100 (moderate: only the data quality suffers), `ROI` = 20, `spws` = 10, `fields` = 1.

The **final score** of a plan is the product of the six lever scores (`n_bin`, `PB`, `cell`, `ROI`, `spws`, `fields`):

* **1**: no mitigation was necessary.
* **0**: the limits cannot be met, even with every mitigation applied.
* In between: the more (and the more drastic, according to the weights) mitigation was needed, the lower.

For `n_bin`, `PB`, `cell` and `ROI` the MOUS-level score is the mean over the kept spws of the per-spw scores.

### Choosing the plan

`enumerate_plans()` lists every plan: which cube levers are applied to each kept spw (at least enough to bring its cube under the cube target), which spws are kept and how many fields are kept. For each plan it computes the mitigated sizes, checks the limits and calculates the lever scores and the final score. Infeasible plans score 0. The feasible plan with the highest score wins; ties go to the plan with fewer levers, then the smaller product size. Cube-lever combinations that can never be part of a better plan are pruned by default (`prune=True`).

The number of plans grows quickly with the number of spws; above `max_candidates` (default 1,000,000) `enumerate_plans()` raises an error.

### Sizes

The sizes are not inputs. They are computed from the imaging target list with `calculate_sizes()`, a port of the original pipeline function (cube size = `4·nx·ny·nchan/nbin/1e9` GB, MFS image and product size as in the pipeline), so the planner can run standalone.

## Quick start

Requires Python >= 3.10 with `pandas`. The notebooks additionally need `numpy`, `matplotlib` and Jupyter.

```python
from smart_size_mitigation_heuristics import Limits, build_mitigation_table, plan

# One dict per science spw: the inputs of the lever formulas.
spws = [
    {"spw": "25", "nchan": 3840, "ratio_satisfied": False,
     "pb": 0.9, "is_mosaic": False, "ppb": 4.0, "robust": 2.0},
    {"spw": "27", "nchan": 960, "ratio_satisfied": True,
     "pb": 1.0, "is_mosaic": False, "ppb": 3.5, "robust": None},
]
nchan = {"25": 3840, "27": 960}

# Imaging targets (in the pipeline: the targets of makeimlist's result).
imlist = [
    {"field": f, "spw": spw, "imsize": (n, n), "nbin": -1, "specmode": "cube"}
    for f in ("f1", "f2", "f3")
    for spw, n in (("25", 1980), ("27", 1400))
]

table = build_mitigation_table(spws)  # size fraction of every lever and spw
limits = Limits(maxcubesize=40.0, maxproductsize=300.0, maxcubelimit=100.0)

final, msg = plan(table, limits, imlist, nchan.__getitem__, repr_spw="25")
print(msg["shortmsg"], round(msg["score"], 3))
print(final[["levers_used", "mitigated_cube_GB", "mitigated_product_GB", "kept"]])
```

`plan()` returns

* `final`: one row per spw with the lever size fractions, the levers applied (`levers_used`), the baseline and mitigated cube and product sizes and a boolean `kept` column.
* `msg`: `shortmsg` / `longmsg` (outcome), `score` (final score), `scores` (score of every lever) and, for very large MOUS, a runtime `warning`. With `return_candidates=True` it also holds `candidates`, the table of all plans sorted by score.

## Repository content

| File | Content |
|------|---------|
| `smart_size_mitigation_heuristics.py` | The whole implementation, see below. Running it directly prints a worked example. |
| `test_mitigation.ipynb` | Minimal end-to-end example: define spws and imaging targets, set limits, run `plan()`, inspect the result and the sorted list of all plans. |
| `diagnostic_plots_mitigation_sizes.ipynb` | Diagnostic plots of the size fraction of every lever (n_bin, PB, hm_cell, ROI, nfields, spws), using the formulas of the module. |
| `diagnostic_plots_mitigation_scores.ipynb` | Diagnostic plots of the lever scores: the score function for the default weights, and for every lever the base score and the weighted score. |

The plot notebooks save their figures as PNGs in `70_results/plots/` (`PLOT_DIR`, defined in the first code cell: one level above the code directory).

### `smart_size_mitigation_heuristics.py`

| Part | Functions |
|------|-----------|
| Sizes | `calculate_sizes()`, `sizes_from_imlist()` |
| Size fractions of the levers | `size_fraction_nbin()`, `size_fraction_pb()`, `size_fraction_cell()`, `size_fraction_roi()`, `size_fraction_fields()`, `size_fraction_spws()`, `minimum_fields_to_remove()` |
| Scores | `LEVER_WEIGHTS`, `lever_score()`, `cube_lever_qualities()` |
| Tables | `build_mitigation_table()`, `add_nfields_columns()`, `add_size_columns()`, `total_size_fraction()` |
| Planning | `Limits`, `enumerate_plans()`, `plan()` |
| Other | `runtime_warning()` (warns about MOUS with many fields and channels; not a mitigation) |

## Known simplifications

These are also listed in the module docstring and are worth resolving before this is implemented into the actual pipeline:

* The PB and cell formulas use a simplified "floor" framing, not the original code's empirical formula, and the carry-over of the converged PB value between steps is not modelled: only *which* levers are used is tracked, not the parameter value each lever converges to. PB, cell and ROI are always applied in full.
* Levers are assumed to compound independently. `n_bin` and `ROI` both act on the spectral axis, so this may not hold.
* The size fraction of a cube lever is applied equally to the cube and to the product size (the product also contains the MFS image, which `n_bin` does not affect).
* The MOUS-level score of `n_bin`, `PB`, `cell` and `ROI` is an unweighted mean over the kept spws.
* The spw lever uses the share of the actual product size; the field lever assumes equal-sized fields.

## Project structure

| Folder | Purpose |
|--------|---------|
| `10_orga/` | Organisation, notes, meeting minutes |
| `20_literature/` | Papers and references |
| `30_data/` | Raw and processed data |
| `50_code/` | Source code (this repo) |
| `70_results/` | Output figures and tables |
| `80_report/` | Report drafts |
| `90_presentation/` | Presentation files |

## Getting started

```bash
git clone https://github.com/lukas-neumann-astro/alma_pipeline_smart_mitigation.git
```
