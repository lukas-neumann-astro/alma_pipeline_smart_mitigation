"""
Score-based size mitigation for the ALMA pipeline's checkproductsize.py.

Instead of the original try -> recompute -> test cascade (nbin, then FoV,
then cell size, then fields, then spws), the mitigation is worked out in
ONE pass from a table of what each lever could achieve:

  1. POTENTIAL   Every lever's possible size reduction is computed up front
                 (mitigation_nbin/pb/cell/roi/fields) and collected in one
                 table, one row per spw, one column per lever.
  2. SCORES      A lever's score is its mitigation value (higher = better).
                 For the aggregate product budget, the per-lever score is
                 the product-size-weighted mean over the spws (lever_scores).
  3. SOLVE       Levers are picked in score order, only as many as needed:
                   - per spw, the cube levers (n_bin, PB, cell, ROI) needed
                     to reach maxcubesize;
                   - then, if the summed product size is still above
                     maxproductsize, the levers are walked down the global
                     score ranking (nfields included) until the budget fits.
  4. SELECT      Only if mitigation alone cannot meet the limits, spws are
                 dropped (representative spw always kept), gated by
                 maxcubelimit and maxproductsize.

The result is a single table (plan()) with one row per spw: the lever
values, which levers were applied, the achieved reduction, the mitigated
cube/product sizes and whether the spw is kept.

Limits:

  maxcubesize    -> per-spw cube target. Soft: never drops a spw by itself.
  maxproductsize -> aggregate product target; mitigation tries to meet it,
                    and spw selection enforces it as a hard budget.
  maxcubelimit   -> per-spw hard gate in the selection: a non-representative
                    spw over this limit is excluded; the representative spw
                    over this limit is an unrecoverable failure.

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
    tracked, not the parameter value each lever converges to.
  - Levers are assumed to compound independently (1 - prod(1 - r_i)); n_bin
    and ROI both act on the spectral axis, so this may not hold.
  - The reduction of the cube-level levers is applied equally to the product
    size and the cube size (the product size also contains the MFS term,
    which n_bin does not affect).
"""

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
    """Field-count analogue of minimum_steps_to_target(). Unlike the
    per-spw levers, there is no menu of distinct levers to rank here --
    mitigation_fields() is a single, deterministic, linear function of how
    many fields are removed -- so the minimum count is solved directly:

        k = ceil(required_reduction * n_fields_total)

    capped at n_fields_total - 1 so the representative field is always
    kept. Returns (n_fields_removed, achieved_reduction)."""
    if required_reduction <= 0 or n_fields_total <= 1:
        return 0, 0.0
    max_removable = n_fields_total - 1
    k = min(math.ceil(required_reduction * n_fields_total), max_removable)
    return k, k / n_fields_total


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


def minimum_steps_to_target(lever_values: dict, required_reduction: float):
    """Given one spw's per-lever mitigation values, rank levers by value
    (higher = better) and apply them one at a time -- compounding, per
    total_reduction()'s logic -- until the cumulative reduction meets
    `required_reduction`. Returns (n_steps, achieved_reduction, levers_used)
    with levers_used in the order applied.

    This answers "what is the fewest levers we need to touch for this spw
    to get it back under the limit", rather than always applying all four
    regardless of whether that's necessary -- so a spw that's only
    slightly over budget might resolve in 1 step, while a badly over-budget
    one may need all 4 (or still fail to reach the target, in which case
    all available levers were applied and the caller has to fall back to
    dropping the spw).

    If a spw isn't over budget at all, required_reduction should be passed
    as 0 (or negative), which returns 0 steps immediately.
    """
    if required_reduction <= 0:
        return 0, 0.0, []

    ordered = sorted(lever_values.items(), key=lambda kv: kv[1], reverse=True)
    remaining = 1.0
    used = []
    for lever, value in ordered:
        if value <= 0:
            break  # sorted descending, so no further lever helps
        remaining *= 1.0 - value
        used.append(lever)
        achieved = 1.0 - remaining
        if achieved >= required_reduction:
            return len(used), achieved, used

    # exhausted all effective levers without reaching the target --
    # caller must fall back to spw filtering for this candidate
    return len(used), 1.0 - remaining, used


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


def add_size_and_step_columns(
    table: pd.DataFrame,
    cube_sizes: dict,
    product_sizes: dict,
    maxcubesize: float,
    columns: list[str] = CUBE_LEVER_COLUMNS,
) -> pd.DataFrame:
    """Extends the lever table with, per spw:
        cube_size_GB, product_size_GB   -- baseline predicted sizes
        required_reduction              -- 0 if already within maxcubesize,
                                            else 1 - maxcubesize/cube_size
        min_steps, levers_used          -- from minimum_steps_to_target()
        achieved_reduction              -- cube-lever reduction actually
                                            reached with those steps (>=
                                            required_reduction unless even
                                            all levers fall short)
        field_reduction                 -- product-only reduction from
                                            dropping fields (0 until applied)
        mitigated_cube_GB, mitigated_product_GB
                                         -- see _refresh_sizes()

    `columns` are the levers that may be used to reach maxcubesize. nfields
    is deliberately not one of them (see the module docstring).

    NOTE: achieved_reduction is applied to product_size_GB as well as
    cube_size_GB, i.e. assuming the product size scales down by the same
    fraction as the cube. That's an approximation -- product size also
    includes the mfs image contribution, which the PB/cell/ROI levers
    affect but n_bin does not -- worth revisiting for precision."""
    table = table.copy()
    table["cube_size_GB"] = [cube_sizes[spw] for spw in table.index]
    table["product_size_GB"] = [product_sizes[spw] for spw in table.index]
    if maxcubesize == -1:  # -1 = no limit (as in the pipeline inputs)
        table["required_reduction"] = 0.0
    else:
        table["required_reduction"] = table.apply(
            lambda r: max(0.0, 1.0 - maxcubesize / r["cube_size_GB"]), axis=1
        )

    steps, achieved, used = [], [], []
    for spw, row in table.iterrows():
        lever_values = {col: row[col] for col in columns}
        n, ach, lv = minimum_steps_to_target(lever_values, row["required_reduction"])
        steps.append(n)
        achieved.append(ach)
        used.append(",".join(lv) if lv else "-")
    table["min_steps"] = steps
    table["levers_used"] = used
    table["achieved_reduction"] = achieved
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
    no reduction anywhere, so it is never picked by minimum_steps_to_target()
    or step2_global_escalation() (both skip levers whose value is <= 0).
    spw_rows entries don't need 'n_roi'/'n_total' keys at all when
    use_roi=False. Pass use_roi=True (and provide n_roi/n_total per spw)
    to switch it on once the real ROI parameter exists in the context.

    nfields: the number of fields/targets of the MOUS -- ONE integer shared
    by all spws (in the real pipeline: len(fields) from the context), not a
    per-spw key. plan() takes the same value as its own `nfields` argument
    and re-applies it (add_nfields_columns), so plan()'s value is the one
    that counts; passing it here only makes the raw table show the nfields
    column. See add_nfields_columns() for what the column holds.

    'nfields' is part of LEVER_COLUMNS, so it takes part in the lever scoring
    and in total_reduction (the POTENTIAL total over all levers). Unlike the
    other levers it only shrinks the product total, never a single cube, so
    the per-spw cube search ignores it (CUBE_LEVER_COLUMNS); plan() decides
    how many fields to drop and writes the achieved value into the
    'nfields' column.
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
# Scores: how much each lever can deliver, for the aggregate product budget
# =============================================================================


def lever_scores(
    table: pd.DataFrame, product_sizes: dict, columns: list[str] = LEVER_COLUMNS
) -> pd.Series:
    """Score of each lever = its mitigation value averaged over the spws,
    weighted by product size (a lever that is strong where the bytes are
    scores higher). Higher is better. For nfields, which has the same
    value in every row, the score is simply its mitigation value.

    The scores are computed once, from the unmitigated table, and the
    ranking they give is not recomputed along the way."""
    weights = pd.Series({spw: product_sizes[spw] for spw in table.index})
    weights = weights / weights.sum()
    scores = {col: float((table[col] * weights).sum()) for col in columns}
    return pd.Series(scores).sort_values(ascending=False)


# =============================================================================
# Solving: apply as few levers as needed, best-scoring first
# =============================================================================


def _apply_cube_lever(table: pd.DataFrame, lever: str) -> pd.DataFrame:
    """Applies one more cube-level lever to every spw that can still use it
    (value > 0 and not applied yet), compounding it into achieved_reduction."""
    table = table.copy()
    for idx, row in table.iterrows():
        used = (
            set(row["levers_used"].split(",")) if row["levers_used"] != "-" else set()
        )
        if lever in used or row[lever] <= 0:
            continue
        table.at[idx, "achieved_reduction"] = 1 - (1 - row["achieved_reduction"]) * (
            1 - row[lever]
        )
        table.at[idx, "levers_used"] = (
            lever if row["levers_used"] == "-" else row["levers_used"] + "," + lever
        )
        table.at[idx, "min_steps"] = row["min_steps"] + 1
    return _refresh_sizes(table)


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


def mitigate(
    table: pd.DataFrame,
    limits: Limits,
    cube_sizes: dict,
    product_sizes: dict,
    nfields: int = 1,
):
    """Works out the mitigation in one pass. Returns (table, scores).

    1. Per spw, the fewest cube levers (best value first) that bring the
       cube under limits.maxcubesize.
    2. If the summed product size is still above limits.maxproductsize, the
       levers are taken in global score order (lever_scores) until it fits.
       nfields takes part in that ranking like any other lever; when it is
       reached, only as many fields are dropped as the budget requires
       (the representative field always stays). Cube levers are applied to
       every spw that can still use them.

    Nothing is dropped here -- see select_spws() for that."""
    table = add_nfields_columns(table, nfields)
    scores = lever_scores(table, product_sizes)
    table = add_size_and_step_columns(
        table, cube_sizes, product_sizes, limits.maxcubesize
    )

    if limits.maxproductsize != -1:
        for lever, score in scores.items():
            total = table["mitigated_product_GB"].sum()
            if total <= limits.maxproductsize or score <= 0:
                break
            if lever == "nfields":
                required = 1.0 - limits.maxproductsize / total
                n_removed, _ = minimum_fields_to_remove(nfields, required)
                if n_removed > 0:
                    table = _apply_fields(table, nfields, n_removed)
            else:
                table = _apply_cube_lever(table, lever)
    return table, scores


# =============================================================================
# Selection: the only place where a spw can be dropped. Gated by maxcubelimit
# (hard) and maxproductsize (budget), on the mitigated sizes.
# =============================================================================


def select_spws(table: pd.DataFrame, limits: Limits, repr_spw: str):
    repr_row = table.loc[repr_spw]
    if (
        limits.maxcubelimit != -1
        and repr_row["mitigated_cube_GB"] > limits.maxcubelimit
    ):
        return None, {
            "longmsg": "Cube size could not be mitigated. The representative spw's "
            "cube size (%.2f GB) is larger than maxcubelimit (%.2f GB)."
            % (repr_row["mitigated_cube_GB"], limits.maxcubelimit),
            "shortmsg": "Cube size could not be mitigated",
        }

    admissible = table[
        (table.index == repr_spw)
        | (limits.maxcubelimit == -1)
        | (table["mitigated_cube_GB"] <= limits.maxcubelimit)
    ].copy()

    max_size = admissible["mitigated_product_GB"].max()
    admissible["score"] = max_size / (admissible["mitigated_product_GB"] + 1e-6)
    admissible = admissible.sort_values("score", ascending=False)
    # representative always first, regardless of score
    admissible = pd.concat([admissible.loc[[repr_spw]], admissible.drop(repr_spw)])

    # oversize cap: at most one non-representative spw above 0.5*maxcubelimit
    kept_rows, oversize_used = [], False
    for spw, row in admissible.iterrows():
        is_oversize = (
            limits.maxcubelimit != -1
            and spw != repr_spw
            and row["mitigated_cube_GB"] > 0.5 * limits.maxcubelimit
        )
        if is_oversize:
            if oversize_used:
                continue
            oversize_used = True
        kept_rows.append(spw)
    admissible = admissible.loc[kept_rows]

    # greedy budget selection (descending score == ascending size, so the
    # break below is a valid knapsack short-circuit)
    selected, used = [], 0.0
    for spw, row in admissible.iterrows():
        if spw == repr_spw:
            selected.append(spw)
            used += row["mitigated_product_GB"]
            continue
        if (
            limits.maxproductsize == -1
            or used + row["mitigated_product_GB"] <= limits.maxproductsize
        ):
            selected.append(spw)
            used += row["mitigated_product_GB"]
        else:
            break

    if limits.maxproductsize != -1 and used > limits.maxproductsize:
        return None, {
            "longmsg": "Product size could not be mitigated. Remaining factor: %.4f."
            % (used / limits.maxproductsize),
            "shortmsg": "Product size could not be mitigated",
        }

    return table.loc[selected], {
        "longmsg": (
            "Size was mitigated"
            if any(table["min_steps"] > 0)
            else "No size mitigation needed"
        ),
        "shortmsg": (
            "Size was mitigated"
            if any(table["min_steps"] > 0)
            else "No size mitigation"
        ),
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
    cube_sizes: dict,
    product_sizes: dict,
    repr_spw: str,
    nfields: int = 1,
    sum_nchan: int | None = None,
):
    """Returns (final_table, msg).

    final_table has one row per spw: the lever values (n_bin, PB, cell,
    ROI, nfields), min_steps / levers_used / achieved_reduction (which
    levers were applied), field_reduction, nfields_kept, the baseline and
    mitigated cube and product sizes, and a boolean 'kept' column. If the
    limits cannot be met (msg['shortmsg'] says so), no spw is kept.

    msg additionally carries 'lever_scores' (the ranking that drove the
    choice) and, if sum_nchan is given and the MOUS is large, 'warning'.

    nfields: number of fields of the MOUS, ONE value for all spws.
    product_sizes[spw]: product size of that spw summed over all fields."""
    table, scores = mitigate(table, limits, cube_sizes, product_sizes, nfields)
    selected, msg = select_spws(table, limits, repr_spw)
    table["kept"] = table.index.isin(selected.index) if selected is not None else False
    msg["lever_scores"] = scores
    if sum_nchan is not None:
        warning = runtime_warning(nfields, sum_nchan)
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

    # Sizes come from calculate_sizes() on an imlist (in the real pipeline
    # this is makeimlist's target list; here: one field, one target per spw).
    example_imlist = [
        {
            "field": "f1",
            "spw": "16 (repr.)",
            "imsize": (2550, 2550),
            "nbin": -1,
            "specmode": "cube",
        },
        {
            "field": "f1",
            "spw": "18",
            "imsize": (2200, 2200),
            "nbin": -1,
            "specmode": "cube",
        },
        {
            "field": "f1",
            "spw": "20",
            "imsize": (1340, 1340),
            "nbin": -1,
            "specmode": "cube",
        },
        {
            "field": "f1",
            "spw": "22",
            "imsize": (2420, 2420),
            "nbin": -1,
            "specmode": "cube",
        },
    ]
    nchan_by_spw = {r["spw"]: r["nchan"] for r in example_spws}
    cube_sizes, max_cube, product_sizes, total_product = calculate_sizes(
        example_imlist, nchan_by_spw.__getitem__
    )
    print(
        f"calculate_sizes: max cube {max_cube:.1f} GB, total product {total_product:.1f} GB"
    )

    final, msg = plan(
        table,
        limits,
        cube_sizes,
        product_sizes,
        repr_spw="16 (repr.)",
        nfields=1,
        sum_nchan=sum(r["nchan"] for r in example_spws),
    )
    print("\nLever scores (product-weighted mitigation, best first):")
    print(msg["lever_scores"].round(3).to_string())
    print(f"\nResult: {msg['shortmsg']}")
    print(final.round(3).to_string())

    # Same MOUS observed with 3 fields: each spw's product size now covers
    # all 3 fields, and nfields competes with the other levers by score.
    product_sizes_3 = {spw: 3 * size for spw, size in product_sizes.items()}
    final3, msg3 = plan(
        table, limits, cube_sizes, product_sizes_3, repr_spw="16 (repr.)", nfields=3
    )
    print("\n--- same MOUS with nfields=3 ---")
    print(msg3["lever_scores"].round(3).to_string())
    print(f"\nResult: {msg3['shortmsg']}")
    print(final3.round(3).to_string())
