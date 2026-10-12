"""
Score-based size mitigation planner for the ALMA pipeline's
checkproductsize.py, combining:

  - the per-lever mitigation formulas and table builder
    (formerly mitigation_table.py)
  - the four-step planner that uses them
    (formerly smart_size_mitigation.py)

into one module, following the four-step structure of the original
mitigate_sizes()/mitigate_imsize() logic:

  Step 1 - per-spw cube size mitigation, gated by maxcubesize (soft).
  Step 2 - aggregate product size mitigation, gated by maxproductsize:
           (a) field reduction, if there is more than one science target
           (b) further GLOBAL lever escalation, if still over budget
               (or if there was only one field to begin with)
  Step 3 - spw selection/filtering, gated by maxcubelimit (hard) and
           maxproductsize (budget) -- this is where a spw actually gets
           dropped, as opposed to Step 1/2 which only shrink cubes.
  Step 4 - large-MOUS runtime warning (unchanged from the original,
           not a mitigation step at all).

Limits map onto this structure the same way they did in the original
code:

  maxcubesize    -> Step 1 only.    Soft: never causes a spw to be
                                     dropped by itself.
  maxproductsize -> Step 2 and 3.   Soft target in Step 2 (more
                                     mitigation is tried), hard budget
                                     in Step 3 (spws get dropped to fit).
  maxcubelimit   -> Step 3 only.    Hard: a non-representative spw over
                                     this limit is excluded outright; the
                                     representative spw over this limit
                                     is an unrecoverable failure.

KNOWN SIMPLIFICATIONS (carried over from earlier in this design, and
worth resolving before this governs real imaging runs):
  - The PB/cell formulas here use a simplified "floor" framing, not the
    original code's empirical exp() formula. In particular, the original
    code carries the PB value Step 1 converged on into Step 2 as a new
    starting floor (PB_mitigation = max(PB_mitigation, PB_limit)) --
    i.e. Step 2 is never allowed to *relax* FoV mitigation below what
    Step 1 already committed to. This planner does not yet model that
    carry-over explicitly; it only tracks *which* levers were used per
    spw, not the specific parameter value each lever converged to.
  - Step 2's field reduction (step2_field_reduction) only reports which
    fields would be kept; the spws belonging to dropped fields are not
    removed from the spw-level table the way a real makeimlist recompute
    would (see apply_field_reduction_to_table()'s docstring).
  - achieved_reduction is applied equally to product_size as to
    cube_size (product size also includes the mfs-image term, which
    n_bin does not affect).
"""

import math
from dataclasses import dataclass
import pandas as pd


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

LEVER_COLUMNS = ["n_bin", "PB", "cell", "ROI", "nfields"]


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
        remaining *= (1.0 - row[col])
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
        remaining *= (1.0 - value)
        used.append(lever)
        achieved = 1.0 - remaining
        if achieved >= required_reduction:
            return len(used), achieved, used

    # exhausted all effective levers without reaching the target --
    # caller must fall back to spw filtering for this candidate
    return len(used), 1.0 - remaining, used


def add_size_and_step_columns(table: pd.DataFrame, cube_sizes: dict, product_sizes: dict,
                               maxcubesize: float, columns: list[str] = LEVER_COLUMNS) -> pd.DataFrame:
    """Extends the lever table with, per spw:
        cube_size_GB, product_size_GB   -- baseline predicted sizes
        required_reduction              -- 0 if already within maxcubesize,
                                            else 1 - maxcubesize/cube_size
        min_steps, levers_used          -- from minimum_steps_to_target()
        achieved_reduction              -- reduction actually reached with
                                            those steps (>= required_reduction
                                            unless even all levers fall short)
        mitigated_cube_GB, mitigated_product_GB
                                         -- baseline sizes scaled by
                                            (1 - achieved_reduction)

    NOTE: achieved_reduction is applied to product_size_GB as well as
    cube_size_GB, i.e. assuming the product size scales down by the same
    fraction as the cube. That's an approximation -- product size also
    includes the mfs image contribution, which the PB/cell/ROI levers
    affect but n_bin does not -- worth revisiting for precision."""
    table = table.copy()
    table["cube_size_GB"] = [cube_sizes[spw] for spw in table.index]
    table["product_size_GB"] = [product_sizes[spw] for spw in table.index]
    table["required_reduction"] = table.apply(
        lambda r: max(0.0, 1.0 - maxcubesize / r["cube_size_GB"]), axis=1)

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

    table["mitigated_cube_GB"] = table["cube_size_GB"] * (1 - table["achieved_reduction"])
    table["mitigated_product_GB"] = table["product_size_GB"] * (1 - table["achieved_reduction"])
    return table


def build_mitigation_table(spw_rows: list[dict], use_roi: bool = False) -> pd.DataFrame:
    """spw_rows: one dict per spw with keys:
        spw, nchan, ratio_satisfied, pb, is_mosaic, ppb, robust, nfields
        and, only if use_roi=True: n_roi, n_total
    Returns a DataFrame indexed by spw with one column per lever plus a
    'total_reduction' column giving the compounded reduction across all
    four CUBE-level levers (see total_reduction() above).

    use_roi: ROI is not yet an implemented pipeline parameter (see
    mitigation_roi()'s docstring), so it is OFF by default. With
    use_roi=False, the ROI column is filled with 0.0 for every spw --
    i.e. present in the table for a consistent shape, but contributing
    no reduction anywhere, so it is never picked by minimum_steps_to_target()
    or step2_global_escalation() (both skip levers whose value is <= 0).
    spw_rows entries don't need 'n_roi'/'n_total' keys at all when
    use_roi=False. Pass use_roi=True (and provide n_roi/n_total per spw)
    to switch it on once the real ROI parameter exists in the context.

    'nfields' is the number of fields/targets this spw is imaged across
    (in the real pipeline, fetched from the context rather than passed by
    hand). Until Step 2's field_reduction logic actually decides how many
    fields to drop, the 'nfields' column here shows the THEORETICAL
    MAXIMUM mitigation from dropping every non-representative field
    (keeping only the representative, i.e. n_fields_kept=1), the same way
    the PB/cell columns show their floor-capped maximum.

    'nfields' is included in LEVER_COLUMNS, so it is treated exactly like
    n_bin/PB/cell/ROI everywhere that reads that list: Step 1's own
    per-spw search (minimum_steps_to_target) may pick it to meet
    maxcubesize, and Step 2b's global escalation may pick it to meet
    maxproductsize, both using this theoretical-max value -- in addition
    to it being overwritten with the real achieved value once Step 2a's
    field_reduction formula runs (apply_field_reduction_to_table()).

    NOTE: this means Step 1/2b can report a spw as already "mitigated via
    nfields" purely on the theoretical-max value, independently of
    whatever Step 2a later actually decides for the aggregate field count
    -- the two are not reconciled. Step 2a's own field_reduction() is
    unaffected (it always runs off the real aggregate product-size budget
    and per-field sizes), so the final, authoritative field count still
    comes from Step 2a; Step 1/2b's use of 'nfields' is best read as "this
    spw could be fully covered by dropping fields alone", not as a
    decision actually taken yet."""
    records = []
    for row in spw_rows:
        roi_value = mitigation_roi(row["n_roi"], row["n_total"]) if use_roi else 0.0
        records.append({
            "spw": row["spw"],
            "n_bin": mitigation_nbin(row["nchan"], row.get("ratio_satisfied", False)),
            "PB": mitigation_pb(row["pb"], row.get("is_mosaic", False)),
            "cell": mitigation_cell(row["ppb"], row.get("robust")),
            "ROI": roi_value,
            "nfields_total": row["nfields"],
            "nfields": mitigation_fields(row["nfields"], 1),
        })
    table = pd.DataFrame(records).set_index("spw")
    table["total_reduction"] = table.apply(total_reduction, axis=1)
    return table


# =============================================================================
# Limits
# =============================================================================

@dataclass
class Limits:
    maxcubesize: float     # GB, per-spw soft target (Step 1)
    maxproductsize: float  # GB, aggregate soft target then hard budget (Step 2/3)
    maxcubelimit: float    # GB, per-spw hard gate (Step 3 only)


# =============================================================================
# Step 2a: field reduction (only relevant when nfields > 1)
# =============================================================================

def step2_field_reduction(fields_product_sizes: dict, repr_field: str, maxproductsize: float):
    """Uses mitigation_fields()/minimum_fields_to_remove() to decide HOW
    MANY fields must be dropped (the uniform-size approximation), then
    uses the actual per-field product sizes to decide WHICH non-
    representative fields to drop: largest-first.

    Dropping the largest fields first is the field-level analogue of
    "apply the highest-mitigation-value lever first" used throughout the
    rest of this planner -- it gets the real (non-uniform) achieved
    reduction to match or exceed the formula's target using no more than
    the formula's predicted field count, rather than, say, dropping
    smallest-first (which would maximize the number of fields kept but
    could require dropping MORE fields than the formula predicted to hit
    the same size target). Worth revisiting if the real priority is
    "keep as many fields as possible" rather than "drop as few as
    possible" -- those are different objectives once field sizes aren't
    actually uniform.

    Returns (kept_fields, kept_total_GB, n_removed, formula_achieved)."""
    total = sum(fields_product_sizes.values())
    if maxproductsize == -1 or total <= maxproductsize:
        return list(fields_product_sizes.keys()), total, 0, 0.0

    n_total = len(fields_product_sizes)
    required_reduction = 1.0 - maxproductsize / total
    k, formula_achieved = minimum_fields_to_remove(n_total, required_reduction)

    others_desc = sorted(
        ((f, s) for f, s in fields_product_sizes.items() if f != repr_field),
        key=lambda kv: kv[1], reverse=True,
    )
    dropped = others_desc[:k]
    kept = [repr_field] + [f for f, _ in others_desc[k:]]
    kept_total = total - sum(s for _, s in dropped)
    return kept, kept_total, k, formula_achieved


# =============================================================================
# Step 2b: global lever escalation (applied uniformly across all spws,
# ranked by aggregate GB saved)
# =============================================================================

def _remaining_levers(levers_used: str, columns=LEVER_COLUMNS):
    used = set(levers_used.split(",")) if levers_used != "-" else set()
    return [c for c in columns if c not in used]


def _apply_additional_lever(table: pd.DataFrame, lever: str, columns=LEVER_COLUMNS) -> pd.DataFrame:
    table = table.copy()
    for idx, row in table.iterrows():
        if lever not in _remaining_levers(row["levers_used"], columns):
            continue
        lever_val = row[lever]
        if lever_val <= 0:
            continue
        new_remaining_frac = (1 - row["achieved_reduction"]) * (1 - lever_val)
        table.at[idx, "achieved_reduction"] = 1 - new_remaining_frac
        table.at[idx, "levers_used"] = (
            row["levers_used"] + "," + lever if row["levers_used"] != "-" else lever
        )
        table.at[idx, "min_steps"] = row["min_steps"] + 1
    table["mitigated_cube_GB"] = table["cube_size_GB"] * (1 - table["achieved_reduction"])
    table["mitigated_product_GB"] = table["product_size_GB"] * (1 - table["achieved_reduction"])
    return table


def step2_global_escalation(table: pd.DataFrame, maxproductsize: float,
                             columns=LEVER_COLUMNS, max_iterations: int = 4) -> pd.DataFrame:
    """Repeatedly picks whichever UNUSED lever would save the most
    aggregate GB across all spws that still have it available, applies it
    to every such spw at once, and rechecks the aggregate budget. This is
    the direct extension of "rank by mitigation value, higher is better"
    to the aggregate/Step-2 problem, instead of the original code's fixed
    nbin -> FoV -> cell sequence."""
    table = table.copy()
    for _ in range(max_iterations):
        total = table["mitigated_product_GB"].sum()
        if maxproductsize == -1 or total <= maxproductsize:
            break

        savings = {}
        for lever in columns:
            saved = 0.0
            for _, row in table.iterrows():
                if lever in _remaining_levers(row["levers_used"], columns) and row[lever] > 0:
                    current_remaining = row["product_size_GB"] * (1 - row["achieved_reduction"])
                    saved += current_remaining * row[lever]
            if saved > 0:
                savings[lever] = saved

        if not savings:
            break  # no lever left that would help anywhere

        best_lever = max(savings, key=savings.get)
        table = _apply_additional_lever(table, best_lever, columns)
    return table


def apply_field_reduction_to_table(table: pd.DataFrame, n_dropped: int) -> pd.DataFrame:
    """Folds the OUTCOME of step2_field_reduction() into the spw-level
    table's bookkeeping, the same way _apply_additional_lever() folds in a
    chosen cube-level lever: updates 'nfields' from its theoretical-max
    placeholder to the actual achieved value, appends 'nfields' to
    levers_used, increments min_steps, compounds it into achieved_reduction,
    and recomputes the mitigated sizes. After this, a spw whose fields got
    cut shows 'nfields' in levers_used in the printed table exactly like
    n_bin does when chosen.

    ASSUMES every spw shares the same global field count and the same
    field-reduction decision (n_dropped applies uniformly across all
    rows) -- true whenever all science-target spws are imaged across the
    same field set, which step2_field_reduction() itself currently
    assumes too (it takes one fields_product_sizes dict, not one per spw).
    If spws ever need independent field counts, this would need to key
    off each row's own 'nfields_total' and a per-row n_dropped instead."""
    if n_dropped <= 0:
        return table

    table = table.copy()
    for idx, row in table.iterrows():
        n_total = row["nfields_total"]
        n_kept = max(1, n_total - n_dropped)
        table.at[idx, "nfields"] = mitigation_fields(n_total, n_kept)
    return _apply_additional_lever(table, "nfields", LEVER_COLUMNS)


# =============================================================================
# Step 3: spw selection, gated by maxcubelimit (hard) and maxproductsize
# (budget), operating on the POST mitigation sizes from steps 1 and 2
# =============================================================================

def step3_select_spws(table: pd.DataFrame, limits: Limits, repr_spw: str):
    repr_row = table.loc[repr_spw]
    if limits.maxcubelimit != -1 and repr_row["mitigated_cube_GB"] > limits.maxcubelimit:
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
        is_oversize = (limits.maxcubelimit != -1 and spw != repr_spw
                        and row["mitigated_cube_GB"] > 0.5 * limits.maxcubelimit)
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
        if limits.maxproductsize == -1 or used + row["mitigated_product_GB"] <= limits.maxproductsize:
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
        "longmsg": "Size was mitigated" if any(table["min_steps"] > 0) else "No size mitigation needed",
        "shortmsg": "Size was mitigated" if any(table["min_steps"] > 0) else "No size mitigation",
    }


# =============================================================================
# Step 4: unchanged runtime warning
# =============================================================================

def step4_runtime_warning(nfields: int, sum_nchan: int,
                           max_num_sciencetargets: int = 30, max_sum_nchan: int = 960) -> str | None:
    if nfields > max_num_sciencetargets and sum_nchan > max_sum_nchan:
        return ("The number of science targets is > %d and the total number of spectral "
                "channels across all science spws > %d. The imaging pipeline will take "
                "substantial time to run on this MOUS." % (max_num_sciencetargets, max_sum_nchan))
    return None


# =============================================================================
# Orchestration
# =============================================================================

def plan(table: pd.DataFrame, limits: Limits, cube_sizes: dict, product_sizes: dict,
         repr_spw: str, nfields: int = 1, fields_product_sizes: dict | None = None,
         repr_field: str | None = None, sum_nchan: int | None = None):

    # Step 1
    table = add_size_and_step_columns(table, cube_sizes, product_sizes, limits.maxcubesize)
    print("-- after Step 1 (per-spw cube mitigation) --")
    print(table[["min_steps", "levers_used", "achieved_reduction",
                  "mitigated_cube_GB", "mitigated_product_GB"]].round(3).to_string())

    # Step 2a: field reduction
    kept_fields = None
    aggregate_for_check = table["mitigated_product_GB"].sum()
    if nfields > 1:
        if fields_product_sizes is None or repr_field is None:
            raise ValueError("fields_product_sizes and repr_field are required when nfields > 1")
        kept_fields, field_total, n_dropped, formula_achieved = step2_field_reduction(
            fields_product_sizes, repr_field, limits.maxproductsize)
        print(f"\n-- Step 2a: field reduction -- dropped {n_dropped} field(s) "
              f"(formula target {formula_achieved:.3f}), kept {kept_fields} "
              f"(product size {field_total:.2f} GB)")
        aggregate_for_check = field_total

        # Fold the outcome into the spw table's own bookkeeping, so
        # 'nfields' shows up in levers_used/achieved_reduction/min_steps
        # for every affected spw, exactly like n_bin does when chosen.
        if n_dropped > 0:
            table = apply_field_reduction_to_table(table, n_dropped)
            print(table[["min_steps", "levers_used", "achieved_reduction",
                          "mitigated_cube_GB", "mitigated_product_GB"]].round(3).to_string())
        # KNOWN GAP: the spws belonging to dropped fields are not removed
        # from `table` -- every spw still contributes its (now field-
        # mitigated) product size below, so Step 2b (if it still runs)
        # escalates levers across ALL spws, including ones whose field was
        # just dropped. Fine for exercising the field-count logic in
        # isolation, but needs the dropped fields' spws excluded from
        # `table` before this chains correctly with a real multi-field
        # dataset.

    # Step 2b: global lever escalation, if the aggregate is still over
    # budget after field reduction (or if there was only one field)
    if limits.maxproductsize != -1 and aggregate_for_check > limits.maxproductsize:
        print(f"\n-- Step 2b: global lever escalation -- "
              f"{aggregate_for_check:.2f} GB > budget {limits.maxproductsize} GB")
        table = step2_global_escalation(table, limits.maxproductsize)
        print(table[["min_steps", "levers_used", "achieved_reduction",
                      "mitigated_cube_GB", "mitigated_product_GB"]].round(3).to_string())

    # Step 3
    selected, msg = step3_select_spws(table, limits, repr_spw)
    print(f"\n-- Step 3: spw selection -- {msg['shortmsg']}")
    if selected is None:
        return None, msg
    print(selected[["min_steps", "levers_used", "achieved_reduction",
                     "mitigated_cube_GB", "mitigated_product_GB"]].round(3).to_string())
    print(f"Total selected product size: {selected['mitigated_product_GB'].sum():.2f} GB")

    # Step 4
    if sum_nchan is not None:
        warning = step4_runtime_warning(nfields, sum_nchan)
        if warning:
            print(f"\n-- Step 4 warning --\n{warning}")

    return selected, msg


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    # Single-field MOUS for this example, so each spw's own "nfields" is 1
    # (matches the plan(nfields=1) call below).
    example_spws = [
        {"spw": "16 (repr.)", "nchan": 3840, "ratio_satisfied": False,
         "pb": 0.9, "is_mosaic": False, "ppb": 4.0, "robust": 2.0, "nfields": 1},
        {"spw": "18", "nchan": 960, "ratio_satisfied": True,
         "pb": 1.0, "is_mosaic": False, "ppb": 3.5, "robust": None, "nfields": 1},
        {"spw": "20", "nchan": 7680, "ratio_satisfied": False,
         "pb": 0.6, "is_mosaic": True, "ppb": 3.0, "robust": None, "nfields": 1},
        {"spw": "22", "nchan": 1920, "ratio_satisfied": False,
         "pb": 0.65, "is_mosaic": False, "ppb": 5.0, "robust": 2.0, "nfields": 1},
    ]

    # ROI is off by default (build_mitigation_table(..., use_roi=True) to
    # turn it on, with n_roi/n_total added to each row above).
    table = build_mitigation_table(example_spws)
    print("Raw per-lever mitigation table (before planning):")
    print(table.round(3).to_string())
    print()

    limits = Limits(maxcubesize=40.0, maxproductsize=150.0, maxcubelimit=60.0)
    cube_sizes = {"16 (repr.)": 100.0, "18": 25.0, "20": 55.0, "22": 45.0}
    product_sizes = {"16 (repr.)": 180.0, "18": 45.0, "20": 99.0, "22": 81.0}

    selected, msg = plan(table, limits, cube_sizes, product_sizes,
                          repr_spw="16 (repr.)", nfields=1,
                          sum_nchan=sum(r["nchan"] for r in example_spws))
