"""
Score-based size mitigation for the ALMA pipeline's checkproductsize.py.

Instead of the original try -> recompute -> test cascade (nbin, then FoV,
then cell size, then fields, then spws), the mitigation is worked out in
ONE pass from a table of what each lever could achieve, and the result is
rated with a single QUALITY SCORE between 0 and 1:

    1    no mitigation was necessary
    0    the limits cannot be met, even with every mitigation applied
    else the more (and the more drastic) mitigation was needed, the lower

Every lever returns its own score, and the final score is their product.
The levers, from most to least moderate (see the score_* functions):

    n_bin, PB, cell   reduce the data quality but remove nothing: 0.9 for
                      n_bin; between 0.9 and 1.0 for PB and cell, the less
                      they change the original values, the closer to 1
    ROI               0.5 to 1.0: 1 if the ROI is the whole spw, down to 0.5
                      when almost everything is cut away
    removing spws     0.4 to 0.6, lower the more spws are removed (the
                      representative spw is always kept)
    removing fields   0.2 to 0.4, lower the more fields are removed (the
                      representative field is always kept)

How the plan is chosen (enumerate_plans() / plan()):

  1. POTENTIAL   The table holds, per spw, what each lever could remove.
  2. ENUMERATE   Every plan is listed: which cube levers (n_bin, PB, cell,
                 ROI) are applied to each kept spw -- at least enough to
                 bring its cube under the cube target -- which spws are kept
                 (the representative one always is) and how many fields are
                 kept (the representative one always is). For each plan the
                 mitigated sizes, the lever scores and the final score are
                 computed.
  3. SELECT      Plans that break a limit (maxcubelimit, the oversize cap,
                 maxproductsize) are infeasible and score 0. The feasible
                 plan with the highest score is the result; ties go to the
                 plan with fewer levers, then the smaller product size.
  The full, sorted list can be returned for diagnostics
  (plan(..., return_candidates=True) -> msg['candidates']).

The result of plan() is a single table with one row per spw (lever values,
levers applied, achieved reduction, mitigated sizes, kept or not, the spw's
own lever score) plus msg['scores'], the score of every lever and the final
score.

Limits:

  maxcubesize    -> per-spw cube target. Soft: never drops a spw by itself.
  maxproductsize -> aggregate product target; mitigation tries to meet it,
                    and spw selection enforces it as a hard budget.
  maxcubelimit   -> per-spw hard gate in the selection: a non-representative
                    spw over this limit is excluded; the representative spw
                    over this limit is an unrecoverable failure (score 0).

nfields is a single, MOUS-wide count (all spws are imaged across the same
fields). It is a PRODUCT-size lever only: dropping fields shrinks the total
product size but not the size of any individual cube, so it never counts
towards maxcubesize. product_sizes[spw] is the product size of that spw
summed over all fields; keeping k of N fields scales it by k/N (equal-sized
fields assumed, the representative field always kept).

KNOWN SIMPLIFICATIONS (worth resolving before this governs real imaging runs):
  - The PB/cell formulas use a simplified "floor" framing, not the original
    code's empirical exp() formula, and the original's carry-over of the
    converged PB value between steps (PB_mitigation = max(PB_mitigation,
    PB_limit)) is not modelled: only WHICH levers are used per spw is
    tracked, not the parameter value each lever converges to. PB, cell and
    ROI are always applied in full (down to the floor / the given ROI).
  - Levers are assumed to compound independently (1 - prod(1 - r_i)); n_bin
    and ROI both act on the spectral axis, so this may not hold.
  - The reduction of the cube-level levers is applied equally to the product
    size and the cube size (the product size also contains the MFS term,
    which n_bin does not affect).
  - The score of n_bin/PB/cell/ROI over the MOUS is the plain mean over the
    kept spws of the per-spw scores (not weighted by size).
"""

import itertools
import math
from dataclasses import dataclass
from typing import Callable
import pandas as pd

# =============================================================================
# Size calculation (ported from CheckProductSizeHeuristics.calculate_sizes)
# =============================================================================


def calculate_sizes(
    imlist: list[dict], get_nchan: Callable[[str], int], full_pol: bool = False
):
    """Same logic as CheckProductSizeHeuristics.calculate_sizes() in
    checkproductsize.py; the only change is that the two lookups it did
    through the pipeline context are passed in instead:

        get_nchan(spw)  replaces
            ref_ms.get_spectral_window(
                context.observing_run.virtual2real_spw_id(int(spw), ref_ms)
            ).num_channels
        full_pol        replaces
            'INTERFEROMETRY_FULL_POL_CUBE_IMAGING' in context.processing_intents

    Inside the pipeline, just keep calling self.calculate_sizes(imlist);
    this port exists so the planner and its examples can run standalone.

    imlist: one dict per imaging target with keys
        field, spw, imsize=(nx, ny), nbin (-1 means 1), specmode.
    Returns (cubesizes, max_cubesize, productsizes, total_productsize):
        cubesizes[spw]    -- max cube size over all targets using that spw
        productsizes[spw] -- product size of the LAST target using that spw
                             (CAS-11363: assumes all targets of a spw are
                             ~the same size)
        total_productsize -- sum of productsize over EVERY target
    All sizes in GB."""
    cubesizes = {}
    productsizes = {}
    total_productsize = 0.0
    for target in imlist:
        nx, ny = target["imsize"]
        nbin = target["nbin"] if target["nbin"] != -1 else 1

        if target["specmode"] == "cube":
            nchan = get_nchan(target["spw"])
            cubesize = 4.0 * nx * ny * nchan / nbin / 1e9
        else:  # 'cont'
            cubesize = 0.0

        # Should include nterms, though overall size is dominated by cube
        # mode, which is currently always nterms=1 (as in the original).
        mfssize = 4.0 * nx * ny / 1e9

        if full_pol:  # original I plus IQUV products
            cubesize = 5 * cubesize
            mfssize = 5 * mfssize

        if cubesizes.get(target["spw"]) is not None:
            cubesizes[target["spw"]] = max(cubesizes[target["spw"]], cubesize)
        else:
            cubesizes[target["spw"]] = cubesize
        productsize = 2.0 * (mfssize + cubesize)
        productsizes[target["spw"]] = productsize
        total_productsize += productsize

    return cubesizes, max(cubesizes.values()), productsizes, total_productsize


def sizes_from_imlist(
    imlist: list[dict], get_nchan: Callable[[str], int], full_pol: bool = False
):
    """Everything the planner needs to know about sizes, from the imaging
    target list alone (calculate_sizes() does the arithmetic):

        cube_sizes[spw]    -- largest cube of that spw over all targets
        product_sizes[spw] -- product size of that spw SUMMED over all its
                              targets, i.e. over all fields (so that the
                              sum over spws equals calculate_sizes()'s
                              total_productsize)
        nfields            -- number of distinct fields in the list

    Returns (cube_sizes, product_sizes, nfields)."""
    cube_sizes = calculate_sizes(imlist, get_nchan, full_pol)[0]
    product_sizes = {
        spw: calculate_sizes(
            [t for t in imlist if t["spw"] == spw], get_nchan, full_pol
        )[3]
        for spw in cube_sizes
    }
    nfields = len({t["field"] for t in imlist})
    return cube_sizes, product_sizes, nfields


# =============================================================================
# Per-lever mitigation formulas
# =============================================================================


def mitigation_nbin(nchan: int, ratio_satisfied: bool = False) -> float:
    """n_bin lever. Eligibility is a lookup, not a continuous function:
    nchan in (7680, 3840) is always eligible for nbin=2; nchan in
    (1920, 960, 480) is only eligible if the channel-width ratio
    condition is met. Reduction is exact: 1 - 1/nbin."""
    if nchan in (7680, 3840):
        nbin = 2
    elif nchan in (1920, 960, 480) and ratio_satisfied:
        nbin = 2
    else:
        nbin = 1
    return 1.0 - 1.0 / nbin


def mitigation_pb(pb: float, is_mosaic: bool = False, floor: float = 0.7) -> float:
    """PB (FoV) lever. Not available for mosaics. For single pointings,
    tightening PB down to `floor` removes 1 - (floor/pb)^2 of the area
    (only positive once pb exceeds the floor)."""
    if is_mosaic:
        return 0.0
    if pb <= floor:
        return 0.0
    return 1.0 - (floor / pb) ** 2


def mitigation_cell(ppb: float, robust: float | None = None) -> float:
    """Cell size (hm_cell) lever. Floor depends on robust: 3.25 ppb for
    robust=2.0 (PIPE-107), else 3.0 ppb. Same area-scaling form as PB."""
    floor = 3.25 if robust == 2.0 else 3.0
    if ppb <= floor:
        return 0.0
    return 1.0 - (floor / ppb) ** 2


def mitigation_roi(n_roi: float, n_total: float) -> float:
    """ROI (channel selection) lever. Not yet implemented in the
    pipeline. Linear in the fraction of channels kept, since it's a
    1-D cut along the spectral axis rather than a 2-D area cut."""
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if n_roi > n_total:
        raise ValueError("n_roi cannot exceed n_total")
    return 1.0 - n_roi / n_total


def mitigation_fields(n_fields_total: int, n_fields_kept: int) -> float:
    """Field/target-count lever (Step 2, aggregate product size -- not a
    per-spw cube-size lever like the four above). Assumes each field
    contributes an equal share of the total product size, so the
    fractional reduction scales directly with the fraction of fields
    dropped: removing 4 of 5 fields removes 4/5 = 0.8 of the size.

    n_fields_kept must be between 1 (only the representative field
    survives -- the theoretical floor, same idea as PB/cell's floor) and
    n_fields_total (nothing dropped)."""
    if n_fields_total <= 0:
        raise ValueError("n_fields_total must be positive")
    if not (1 <= n_fields_kept <= n_fields_total):
        raise ValueError("n_fields_kept must be between 1 and n_fields_total")
    return (n_fields_total - n_fields_kept) / n_fields_total


def minimum_fields_to_remove(n_fields_total: int, required_reduction: float):
    """Fewest fields to remove for a required product reduction. There is
    no menu of levers to choose from here -- mitigation_fields() is a
    single, linear function of how many fields are removed -- so the count
    is solved directly:

        k = ceil(required_reduction * n_fields_total)

    capped at n_fields_total - 1 so the representative field is always
    kept. Returns (n_fields_removed, achieved_reduction)."""
    if required_reduction <= 0 or n_fields_total <= 1:
        return 0, 0.0
    max_removable = n_fields_total - 1
    k = min(math.ceil(required_reduction * n_fields_total), max_removable)
    return k, k / n_fields_total


# =============================================================================
# Quality scores: how much a mitigation hurts (1 = not at all, 0 = worst)
# =============================================================================

SCORE_NBIN = 0.9  # channel binning, when applied
SCORE_MIN_PB_CELL = 0.9  # PB / cell at a mitigation of 100 % (never reached)
SCORE_MIN_ROI = 0.5  # ROI that cuts away (almost) the whole spw
SCORE_RANGE_SPWS = (
    0.4,
    0.6,
)  # (worst, best): all but the repr. spw / almost none removed
SCORE_RANGE_FIELDS = (
    0.2,
    0.4,
)  # (worst, best): all but the repr. field / almost none removed


def score_nbin() -> float:
    """Channel binning: a fixed 0.9 whenever it is applied."""
    return SCORE_NBIN


def score_pb(mitigation: float) -> float:
    """PB (FoV) tightening: 0.9 to 1.0, the closer to 1 the less the FoV
    changes. `mitigation` is the size reduction from mitigation_pb()."""
    return 1.0 - (1.0 - SCORE_MIN_PB_CELL) * mitigation


def score_cell(mitigation: float) -> float:
    """Cell size (hm_cell) increase: 0.9 to 1.0, like score_pb()."""
    return 1.0 - (1.0 - SCORE_MIN_PB_CELL) * mitigation


def score_roi(mitigation: float) -> float:
    """ROI: 1.0 for the whole spw (mitigation 0), falling linearly to 0.5
    when everything is cut away (mitigation 1). `mitigation` is
    mitigation_roi(), the removed fraction."""
    return 1.0 - (1.0 - SCORE_MIN_ROI) * mitigation


def _removal_score(n_total: int, n_removed: int, score_range: tuple) -> float:
    worst, best = score_range
    max_removable = n_total - 1  # the representative one always stays
    if n_removed <= 0 or max_removable <= 0:
        return 1.0
    n_removed = min(n_removed, max_removable)
    return best - (best - worst) * n_removed / max_removable


def score_spws(n_total: int, n_removed: int) -> float:
    """Removing spws: 1.0 if none; otherwise 0.6 (almost none removed)
    falling linearly to 0.4 (all but the representative spw removed)."""
    return _removal_score(n_total, n_removed, SCORE_RANGE_SPWS)


def score_fields(n_total: int, n_removed: int) -> float:
    """Removing fields: 1.0 if none; otherwise 0.4 (almost none removed)
    falling linearly to 0.2 (all but the representative field removed)."""
    return _removal_score(n_total, n_removed, SCORE_RANGE_FIELDS)


def cube_lever_qualities(table: pd.DataFrame) -> pd.DataFrame:
    """Score each cube lever WOULD have if applied to each spw (rows: spws,
    columns: CUBE_LEVER_COLUMNS). Levers that cannot be applied (value 0)
    get 1.0."""
    q = pd.DataFrame(index=table.index)
    q["n_bin"] = [SCORE_NBIN if m > 0 else 1.0 for m in table["n_bin"]]
    q["PB"] = [score_pb(m) for m in table["PB"]]
    q["cell"] = [score_cell(m) for m in table["cell"]]
    q["ROI"] = [score_roi(m) for m in table["ROI"]]
    return q


# =============================================================================
# Mitigation table builder
# =============================================================================

# Levers that shrink an individual cube (and therefore also the products).
CUBE_LEVER_COLUMNS = ["n_bin", "PB", "cell", "ROI"]
# All levers. nfields is the odd one out: it only shrinks the product total.
LEVER_COLUMNS = CUBE_LEVER_COLUMNS + ["nfields"]


def total_reduction(row: pd.Series, columns: list[str] = LEVER_COLUMNS) -> float:
    """Combined fractional size reduction if all levers in `columns` were
    applied together, assuming they act independently/multiplicatively on
    the remaining size: applying reductions r_1, r_2, ... in sequence
    leaves a fraction (1-r_1)*(1-r_2)*... of the original size, so the
    total reduction is 1 minus that product.

    NOTE: this treats the four levers as compounding independently, which
    is a simplifying assumption -- e.g. n_bin (channels) and ROI (also
    channels) both act along the spectral axis, so applying both is not
    obviously independent the way n_bin and PB (spectral vs spatial) are.
    Worth revisiting once the real interaction between levers is known."""
    remaining = 1.0
    for col in columns:
        remaining *= 1.0 - row[col]
    return 1.0 - remaining


def _refresh_sizes(table: pd.DataFrame) -> pd.DataFrame:
    """Recomputes the mitigated sizes from the cube-lever reduction
    (achieved_reduction, hits cube AND product) and the field reduction
    (field_reduction, hits the product only)."""
    table["mitigated_cube_GB"] = table["cube_size_GB"] * (
        1 - table["achieved_reduction"]
    )
    table["mitigated_product_GB"] = (
        table["product_size_GB"]
        * (1 - table["achieved_reduction"])
        * (1 - table["field_reduction"])
    )
    return table


def add_size_columns(
    table: pd.DataFrame,
    cube_sizes: dict,
    product_sizes: dict,
    maxcubesize: float,
    maxcubelimit: float = -1,
) -> pd.DataFrame:
    """Extends the lever table with, per spw:
        cube_size_GB, product_size_GB   -- baseline predicted sizes
        required_reduction              -- 0 if already within the cube target
                                            (the smaller of maxcubesize and
                                            maxcubelimit, -1 = unused), else
                                            1 - target/cube_size
        min_steps, levers_used, achieved_reduction, field_reduction
                                         -- nothing applied yet (0, '-', 0, 0);
                                            filled in for the chosen plan
        mitigated_cube_GB, mitigated_product_GB
                                         -- see _refresh_sizes()

    NOTE: achieved_reduction is applied to product_size_GB as well as
    cube_size_GB, i.e. assuming the product size scales down by the same
    fraction as the cube. That's an approximation -- product size also
    includes the mfs image contribution, which the PB/cell/ROI levers
    affect but n_bin does not -- worth revisiting for precision."""
    table = table.copy()
    table["cube_size_GB"] = [cube_sizes[spw] for spw in table.index]
    table["product_size_GB"] = [product_sizes[spw] for spw in table.index]
    targets = [x for x in (maxcubesize, maxcubelimit) if x != -1]  # -1 = no limit
    if targets:
        target = min(targets)
        table["required_reduction"] = table["cube_size_GB"].map(
            lambda size: max(0.0, 1.0 - target / size)
        )
    else:
        table["required_reduction"] = 0.0
    table["min_steps"] = 0
    table["levers_used"] = "-"
    table["achieved_reduction"] = 0.0
    table["field_reduction"] = 0.0
    return _refresh_sizes(table)


def add_nfields_columns(table: pd.DataFrame, nfields: int) -> pd.DataFrame:
    """Adds/overwrites the field-count columns for a MOUS-wide field count.

    nfields is ONE integer for the whole MOUS (every science spw is imaged
    across the same set of fields), not a per-spw property. It is applied
    identically to every row:
        nfields_total -- the field count itself
        nfields       -- the THEORETICAL MAXIMUM mitigation from dropping
                         every non-representative field (n_fields_kept=1),
                         the same way the PB/cell columns show their
                         floor-capped maximum
    total_reduction is recomputed, since it compounds over LEVER_COLUMNS,
    which includes 'nfields'."""
    if nfields < 1:
        raise ValueError("nfields must be >= 1")
    table = table.copy()
    table["nfields_total"] = nfields
    table["nfields_kept"] = nfields
    table["nfields"] = mitigation_fields(nfields, 1)
    table["total_reduction"] = table.apply(total_reduction, axis=1)
    return table


def build_mitigation_table(
    spw_rows: list[dict], use_roi: bool = False, nfields: int = 1
) -> pd.DataFrame:
    """spw_rows: one dict per spw with keys:
        spw, nchan, ratio_satisfied, pb, is_mosaic, ppb, robust
        and, only if use_roi=True: n_roi, n_total
    Returns a DataFrame indexed by spw with one column per lever plus a
    'total_reduction' column giving the compounded reduction across all
    levers in LEVER_COLUMNS (see total_reduction() above).

    use_roi: ROI is not yet an implemented pipeline parameter (see
    mitigation_roi()'s docstring), so it is OFF by default. With
    use_roi=False, the ROI column is filled with 0.0 for every spw --
    i.e. present in the table for a consistent shape, but contributing
    no reduction anywhere, so it is never applied (levers whose value is <= 0 are skipped).
    spw_rows entries don't need 'n_roi'/'n_total' keys at all when
    use_roi=False. Pass use_roi=True (and provide n_roi/n_total per spw)
    to switch it on once the real ROI parameter exists in the context.

    nfields: the number of fields/targets of the MOUS -- ONE integer shared
    by all spws (in the real pipeline: len(fields) from the context), not a
    per-spw key. plan() takes the same value as its own `nfields` argument
    and re-applies it (add_nfields_columns), so plan()'s value is the one
    that counts; passing it here only makes the raw table show the nfields
    column. See add_nfields_columns() for what the column holds.

    'nfields' is part of LEVER_COLUMNS, so it takes part in total_reduction
    (the POTENTIAL total over all levers). Unlike the other levers it only
    shrinks the product total, never a single cube, so the per-spw cube
    search ignores it (CUBE_LEVER_COLUMNS); plan() decides how many fields
    to drop and writes the achieved value into the 'nfields' column.
    """
    records = []
    for row in spw_rows:
        roi_value = mitigation_roi(row["n_roi"], row["n_total"]) if use_roi else 0.0
        records.append(
            {
                "spw": row["spw"],
                "n_bin": mitigation_nbin(
                    row["nchan"], row.get("ratio_satisfied", False)
                ),
                "PB": mitigation_pb(row["pb"], row.get("is_mosaic", False)),
                "cell": mitigation_cell(row["ppb"], row.get("robust")),
                "ROI": roi_value,
            }
        )
    table = pd.DataFrame(records).set_index("spw")
    return add_nfields_columns(table, nfields)


# =============================================================================
# Limits
# =============================================================================


@dataclass
class Limits:
    maxcubesize: float  # GB, per-spw soft target (Step 1)
    maxproductsize: float  # GB, aggregate soft target then hard budget (Step 2/3)
    maxcubelimit: float  # GB, per-spw hard gate (Step 3 only)


# =============================================================================
# Enumerating every plan
# =============================================================================


def _lever_subsets(available: list[str]):
    for k in range(len(available) + 1):
        yield from itertools.combinations(available, k)


def _spw_options(row: pd.Series, qualities: pd.Series, prune: bool) -> list[dict]:
    """All sensible cube-lever combinations for one spw.

    A combination is sensible if it brings the cube under the cube target
    (required_reduction); if no combination can, only the one applying every
    available lever is (the cube is then as small as it can get). With
    prune=True, a combination is dropped when another one reduces at least
    as much AND scores at least as well on every lever -- it can never be
    part of a better plan, so this loses nothing but diagnostic rows."""
    available = [lv for lv in CUBE_LEVER_COLUMNS if row[lv] > 0]
    options = []
    for subset in _lever_subsets(available):
        remaining = 1.0
        for lv in subset:
            remaining *= 1.0 - row[lv]
        options.append(
            {
                "levers": subset,
                "reduction": 1.0 - remaining,
                "q": tuple(
                    qualities[lv] if lv in subset else 1.0 for lv in CUBE_LEVER_COLUMNS
                ),
            }
        )
    reaching = [
        o for o in options if o["reduction"] >= row["required_reduction"] - 1e-12
    ]
    options = reaching or [max(options, key=lambda o: o["reduction"])]
    if prune:
        options = [
            o
            for i, o in enumerate(options)
            if not any(
                p["reduction"] >= o["reduction"]
                and all(a >= b for a, b in zip(p["q"], o["q"]))
                and (p["reduction"] > o["reduction"] or p["q"] != o["q"] or j < i)
                for j, p in enumerate(options)
                if j != i
            )
        ]
    return options


def enumerate_plans(
    table: pd.DataFrame,
    limits: Limits,
    repr_spw: str,
    nfields: int = 1,
    prune: bool = True,
    max_candidates: int = 1_000_000,
) -> pd.DataFrame:
    """Every plan -- a choice of cube levers per kept spw, of which spws to
    keep (the representative one always is) and of how many fields to keep
    (the representative field always is) -- with its sizes, its lever scores
    and its final score. `table` must already hold the size columns
    (add_size_columns()). One row per plan, best first.

    Columns:
      <one per spw>   cube levers applied to that spw ('dropped' if removed)
      n_spws_kept, nfields_kept, n_levers
      product_GB, max_cube_GB   summed product size / largest cube of the
                                kept spws after mitigation
      feasible        all limits met (cube limit, oversize cap, product budget)
      n_bin, PB, cell, ROI, spws, fields   the lever scores
      score           their product, or 0 if not feasible
    Sorted by score (best first), then fewer levers, then smaller product.

    prune: leave out cube-lever combinations that can never be part of a
    better plan (see _spw_options); False lists literally every one.
    The number of plans grows quickly with the number of spws, so more than
    max_candidates raises an error."""
    spws = [repr_spw] + [s for s in table.index if s != repr_spw]
    qualities = cube_lever_qualities(table)
    options = {s: _spw_options(table.loc[s], qualities.loc[s], prune) for s in spws}

    n_plans = nfields * len(options[repr_spw])
    for s in spws[1:]:
        n_plans *= 1 + len(options[s])
    if n_plans > max_candidates:
        raise ValueError(
            f"{n_plans} plans to enumerate (> max_candidates={max_candidates}); "
            "raise max_candidates, or use prune=True"
        )

    limit = limits.maxcubelimit
    budget = limits.maxproductsize
    n_spw = len(spws)
    rows = []
    for mask in range(2 ** (n_spw - 1)):
        kept = [spws[0]] + [s for i, s in enumerate(spws[1:]) if mask >> i & 1]
        for combo in itertools.product(*(options[s] for s in kept)):
            cubes = [
                table.at[s, "cube_size_GB"] * (1 - o["reduction"])
                for s, o in zip(kept, combo)
            ]
            product0 = sum(
                table.at[s, "product_size_GB"] * (1 - o["reduction"])
                for s, o in zip(kept, combo)
            )
            cube_ok = limit == -1 or (
                cubes[0] <= limit + 1e-9
                and all(c <= limit + 1e-9 for c in cubes[1:])
                and sum(c > 0.5 * limit for c in cubes[1:]) <= 1
            )
            lever_scores = [
                sum(o["q"][i] for o in combo) / len(kept)
                for i in range(len(CUBE_LEVER_COLUMNS))
            ]
            spw_score = score_spws(n_spw, n_spw - len(kept))
            levers = {s: ",".join(o["levers"]) or "-" for s, o in zip(kept, combo)}
            n_levers = sum(len(o["levers"]) for o in combo)
            for n_kept in range(1, nfields + 1):
                product = product0 * n_kept / nfields
                feasible = cube_ok and (budget == -1 or product <= budget + 1e-9)
                field_score = score_fields(nfields, nfields - n_kept)
                score = math.prod(lever_scores) * spw_score * field_score
                rows.append(
                    [levers.get(s, "dropped") for s in spws]
                    + [len(kept), n_kept, n_levers, product, max(cubes), feasible]
                    + lever_scores
                    + [spw_score, field_score, score if feasible else 0.0]
                )
    columns = (
        list(spws)
        + ["n_spws_kept", "nfields_kept", "n_levers", "product_GB", "max_cube_GB"]
        + ["feasible"]
        + CUBE_LEVER_COLUMNS
        + ["spws", "fields", "score"]
    )
    plans = pd.DataFrame(rows, columns=columns)
    return plans.sort_values(
        ["score", "n_levers", "product_GB"], ascending=[False, True, True]
    ).reset_index(drop=True)


def _apply_plan(table: pd.DataFrame, best: pd.Series, spws: list[str], nfields: int):
    """The per-spw result table for the chosen plan: levers applied to the
    kept spws, fields removed, mitigated sizes, and the 'kept' column."""
    table = table.copy()
    for spw in spws:
        used = best[spw]
        if used == "dropped":
            continue
        levers = [] if used == "-" else used.split(",")
        remaining = 1.0
        for lever in levers:
            remaining *= 1.0 - table.at[spw, lever]
        table.at[spw, "achieved_reduction"] = 1.0 - remaining
        table.at[spw, "levers_used"] = used
        table.at[spw, "min_steps"] = len(levers)
    table["kept"] = [best[spw] != "dropped" for spw in table.index]
    table = _refresh_sizes(table)
    n_removed = nfields - int(best["nfields_kept"])
    if n_removed > 0:
        table = _apply_fields(table, nfields, n_removed)
    return table


def _apply_fields(table: pd.DataFrame, n_fields: int, n_removed: int) -> pd.DataFrame:
    """Drops n_removed of the n_fields fields (the representative field is
    always among the kept ones). Product-size lever only: field_reduction
    hits mitigated_product_GB, not mitigated_cube_GB. The 'nfields' column
    changes from its theoretical maximum to the value actually achieved."""
    table = table.copy()
    reduction = mitigation_fields(n_fields, n_fields - n_removed)
    table["nfields"] = reduction
    table["field_reduction"] = reduction
    table["nfields_kept"] = n_fields - n_removed
    table["levers_used"] = [
        "nfields" if used == "-" else used + ",nfields" for used in table["levers_used"]
    ]
    table["min_steps"] = table["min_steps"] + 1
    return _refresh_sizes(table)


def _failure_message(table: pd.DataFrame, limits: Limits, repr_spw: str) -> dict:
    """Why no plan is feasible: the representative spw's cube cannot be
    brought under maxcubelimit, or the product budget cannot be met."""
    cube_floor = table.at[repr_spw, "cube_size_GB"] * (
        1 - total_reduction(table.loc[repr_spw], CUBE_LEVER_COLUMNS)
    )
    if limits.maxcubelimit != -1 and cube_floor > limits.maxcubelimit:
        return {
            "longmsg": "Cube size could not be mitigated. The representative spw's "
            "smallest possible cube size (%.2f GB) is larger than maxcubelimit "
            "(%.2f GB)." % (cube_floor, limits.maxcubelimit),
            "shortmsg": "Cube size could not be mitigated",
            "failure": "cube",
        }
    smallest = (
        1 - total_reduction(table.loc[repr_spw], CUBE_LEVER_COLUMNS)
    ) / table.at[repr_spw, "nfields_total"]
    product_floor = table.at[repr_spw, "product_size_GB"] * smallest
    return {
        "longmsg": "Product size could not be mitigated. Remaining factor: %.4f."
        % (product_floor / limits.maxproductsize),
        "shortmsg": "Product size could not be mitigated",
        "failure": "product",
    }


# =============================================================================
# Runtime warning for large MOUS (not a mitigation)
# =============================================================================


def runtime_warning(
    nfields: int,
    sum_nchan: int,
    max_num_sciencetargets: int = 30,
    max_sum_nchan: int = 960,
) -> str | None:
    if nfields > max_num_sciencetargets and sum_nchan > max_sum_nchan:
        return (
            "The number of science targets is > %d and the total number of spectral "
            "channels across all science spws > %d. The imaging pipeline will take "
            "substantial time to run on this MOUS."
            % (max_num_sciencetargets, max_sum_nchan)
        )
    return None


# =============================================================================
# plan(): the one-call entry point
# =============================================================================


def plan(
    table: pd.DataFrame,
    limits: Limits,
    imlist: list[dict],
    get_nchan: Callable[[str], int],
    repr_spw: str,
    full_pol: bool = False,
    nfields: int | None = None,
    return_candidates: bool = False,
    prune: bool = True,
):
    """Returns (final_table, msg).

    The sizes are not inputs: they are computed from the imaging target
    list with calculate_sizes() (see sizes_from_imlist()), exactly as the
    pipeline does for its own default imaging setup.

      imlist     one dict per imaging target: field, spw, imsize=(nx, ny),
                 nbin (-1 = 1), specmode. In the pipeline: the targets of
                 makeimlist's result.
      get_nchan  spw -> number of channels (in the pipeline: from the
                 reference MS).
      full_pol   True for full-pol cube imaging (sizes x5).
      nfields    number of fields, ONE value for all spws. Default: the
                 number of distinct fields in imlist.
      return_candidates
                 also return the list of ALL plans, best first, in
                 msg['candidates'] (see enumerate_plans()).
      prune      passed to enumerate_plans().

    The spws in `table` must be the spws of imlist.

    Every plan is enumerated and scored (enumerate_plans()); the plan with
    the highest score is returned.

    final_table has one row per spw: the lever values (n_bin, PB, cell,
    ROI, nfields), min_steps / levers_used / achieved_reduction (which
    levers were applied), field_reduction, nfields_kept, the baseline and
    mitigated cube and product sizes and a boolean 'kept' column. If the
    limits cannot be met (msg['shortmsg'] says so), no spw is kept.

    msg['scores'] holds the score of every lever for the chosen plan
    (n_bin, PB, cell, ROI, spws, fields; 1.0 = lever not used) and
    msg['score'] their product, the final score in [0, 1]: 1 = no
    mitigation necessary, 0 = the limits cannot be met. If the MOUS is
    large, msg['warning']."""
    cube_sizes, product_sizes, n_fields_in_list = sizes_from_imlist(
        imlist, get_nchan, full_pol
    )
    if set(table.index) != set(cube_sizes):
        raise ValueError(
            f"table spws {sorted(table.index)} do not match imlist spws "
            f"{sorted(cube_sizes)}"
        )
    if nfields is None:
        nfields = n_fields_in_list

    table = add_nfields_columns(table, nfields)
    table = add_size_columns(
        table, cube_sizes, product_sizes, limits.maxcubesize, limits.maxcubelimit
    )
    plans = enumerate_plans(table, limits, repr_spw, nfields, prune)
    best = plans.iloc[0]

    if best["feasible"]:
        spws = [c for c in plans.columns[: len(table)]]
        table = _apply_plan(table, best, spws, nfields)
        mitigated = best["n_levers"] > 0 or best["n_spws_kept"] < len(table)
        mitigated = mitigated or best["nfields_kept"] < nfields
        msg = {
            "longmsg": (
                "Size was mitigated" if mitigated else "No size mitigation needed"
            ),
            "shortmsg": "Size was mitigated" if mitigated else "No size mitigation",
        }
        msg["scores"] = best[CUBE_LEVER_COLUMNS + ["spws", "fields"]].astype(float)
        msg["score"] = float(best["score"])
    else:
        table["kept"] = False
        msg = _failure_message(table, limits, repr_spw)
        msg["scores"] = pd.Series(
            float("nan"), index=CUBE_LEVER_COLUMNS + ["spws", "fields"]
        )
        msg["score"] = 0.0
    if return_candidates:
        msg["candidates"] = plans
    warning = runtime_warning(nfields, sum(get_nchan(spw) for spw in cube_sizes))
    if warning:
        msg["warning"] = warning
    return table, msg


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    # Single-field MOUS for this example: nfields=1 is passed once, to
    # plan(), for the whole MOUS (not per spw).
    example_spws = [
        {
            "spw": "16 (repr.)",
            "nchan": 3840,
            "ratio_satisfied": False,
            "pb": 0.9,
            "is_mosaic": False,
            "ppb": 4.0,
            "robust": 2.0,
        },
        {
            "spw": "18",
            "nchan": 960,
            "ratio_satisfied": True,
            "pb": 1.0,
            "is_mosaic": False,
            "ppb": 3.5,
            "robust": None,
        },
        {
            "spw": "20",
            "nchan": 7680,
            "ratio_satisfied": False,
            "pb": 0.6,
            "is_mosaic": True,
            "ppb": 3.0,
            "robust": None,
        },
        {
            "spw": "22",
            "nchan": 1920,
            "ratio_satisfied": False,
            "pb": 0.65,
            "is_mosaic": False,
            "ppb": 5.0,
            "robust": 2.0,
        },
    ]

    # ROI is off by default (build_mitigation_table(..., use_roi=True) to
    # turn it on, with n_roi/n_total added to each row above).
    table = build_mitigation_table(example_spws)
    print("Raw per-lever mitigation table (before planning):")
    print(table.round(3).to_string())
    print()

    limits = Limits(maxcubesize=40.0, maxproductsize=150.0, maxcubelimit=60.0)

    # The sizes are computed from the imaging target list (in the pipeline:
    # makeimlist's targets) -- here one field with one target per spw.
    imsizes = {"16 (repr.)": 2550, "18": 2200, "20": 1340, "22": 2420}
    nchan_by_spw = {r["spw"]: r["nchan"] for r in example_spws}
    example_imlist = [
        {"field": "f1", "spw": spw, "imsize": (n, n), "nbin": -1, "specmode": "cube"}
        for spw, n in imsizes.items()
    ]
    final, msg = plan(
        table, limits, example_imlist, nchan_by_spw.__getitem__, repr_spw="16 (repr.)"
    )
    print(f"\nResult: {msg['shortmsg']}  (final score {msg['score']:.3f})")
    print(msg["scores"].round(3).to_string())
    print(final.round(3).to_string())

    # Same MOUS observed with 3 fields: the number of fields is read from
    # the imlist; fields are only dropped as a last resort.
    imlist_3 = [dict(t, field=f) for f in ("f1", "f2", "f3") for t in example_imlist]
    final3, msg3 = plan(
        table, limits, imlist_3, nchan_by_spw.__getitem__, repr_spw="16 (repr.)"
    )
    print("\n--- same MOUS with 3 fields ---")
    print(f"\nResult: {msg3['shortmsg']}  (final score {msg3['score']:.3f})")
    print(msg3["scores"].round(3).to_string())
    print(final3.round(3).to_string())
