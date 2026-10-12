"""
Plots for each individual mitigation lever defined in size_mitigation.py:
n_bin, PB (field of view), cell (hm_cell), ROI (region of interest, not
yet implemented in the real pipeline), and nfields (field/target count).

All formulas are imported directly from size_mitigation.py rather than
reimplemented here, so these plots always match whatever the planner
actually uses.
"""

import matplotlib.pyplot as plt
import numpy as np

from size_mitigation import (
    mitigation_nbin,
    mitigation_pb,
    mitigation_cell,
    mitigation_roi,
    mitigation_fields,
)

# =============================================================================
# n_bin mitigation
# =============================================================================

# The five nchan values the eligibility rule treats specially.
eligible_nchans = [480, 960, 1920, 3840, 7680]

mitigation_ratio_false = [mitigation_nbin(n, ratio_satisfied=False) for n in eligible_nchans]
mitigation_ratio_true = [mitigation_nbin(n, ratio_satisfied=True) for n in eligible_nchans]

print(f"{'nchan':>8} {'mitigation (ratio unmet)':>26} {'mitigation (ratio met)':>24}")
for n, m_false, m_true in zip(eligible_nchans, mitigation_ratio_false, mitigation_ratio_true):
    print(f"{n:>8} {m_false:>26.3f} {m_true:>24.3f}")

fig, ax = plt.subplots(figsize=(5, 5/1.4))
ax.plot(eligible_nchans, mitigation_ratio_false, marker="o",
        label="online channel averaging (1920/960/480 only)", color="#4C72B0")
ax.plot(eligible_nchans, mitigation_ratio_true, marker="s", linestyle="--",
        label="no online channel averaging (1920/960/480 only)", color="#DD8452")
ax.set_xticks(eligible_nchans)
ax.set_xlabel("nchan")
ax.set_ylabel("Fractional size reduction")
ax.set_title("n_bin mitigation")
ax.set_ylim(0, 1)
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("nbin_mitigation.png", dpi=150)
print("\nSaved plot to nbin_mitigation.png")


# =============================================================================
# PB mitigation
# =============================================================================

PB_FLOOR = 0.7

pb_values = np.linspace(0.3, 1.0, 300)
mitigation_single_pointing = [mitigation_pb(pb, is_mosaic=False) for pb in pb_values]
mitigation_mosaic = [mitigation_pb(pb, is_mosaic=True) for pb in pb_values]

sample_pbs = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
print(f"\n{'PB':>6} {'mitigation (single pointing)':>30} {'mitigation (mosaic)':>22}")
for pb in sample_pbs:
    print(f"{pb:>6.2f} {mitigation_pb(pb, is_mosaic=False):>30.3f} "
          f"{mitigation_pb(pb, is_mosaic=True):>22.3f}")

fig, ax = plt.subplots(figsize=(5, 5/1.4))
ax.plot(pb_values, mitigation_single_pointing, label="single pointing", color="#4C72B0")
ax.plot(pb_values, mitigation_mosaic, label="mosaic (never mitigated)",
        color="#DD8452", linestyle="--")
ax.axvline(PB_FLOOR, color="gray", linewidth=0.8, linestyle=":")
ax.annotate(f"PB floor = {PB_FLOOR}", xy=(PB_FLOOR, 0.02), xytext=(PB_FLOOR + 0.03, 0.08),
            fontsize=9, color="gray")
ax.set_xlabel("PB")
ax.set_ylabel("Fractional size reduction")
ax.set_title("PB mitigation")
ax.set_ylim(-0.02, 0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("pb_mitigation.png", dpi=150)
print("\nSaved plot to pb_mitigation.png")


# =============================================================================
# hm_cell mitigation
# =============================================================================

CELL_FLOOR_ROBUST_2 = 3.25
CELL_FLOOR_DEFAULT = 3.0

ppb_values = np.linspace(2.5, 6.0, 300)
mitigation_robust2 = [mitigation_cell(ppb, robust=2.0) for ppb in ppb_values]
mitigation_default = [mitigation_cell(ppb, robust=None) for ppb in ppb_values]

sample_ppbs = [2.5, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0]
print(f"\n{'ppb':>6} {'mitigation (robust=2.0)':>26} {'mitigation (other robust)':>28}")
for ppb in sample_ppbs:
    print(f"{ppb:>6.2f} {mitigation_cell(ppb, robust=2.0):>26.3f} "
          f"{mitigation_cell(ppb, robust=None):>28.3f}")

fig, ax = plt.subplots(figsize=(5, 5/1.4))
ax.plot(ppb_values, mitigation_robust2, label="robust = 2.0 (floor 3.25 ppb)", color="#4C72B0")
ax.plot(ppb_values, mitigation_default, label="robust != 2.0 (floor 3.0 ppb)",
        color="#DD8452", linestyle="--")
ax.axvline(CELL_FLOOR_ROBUST_2, color="#4C72B0", linewidth=0.8, linestyle=":")
ax.axvline(CELL_FLOOR_DEFAULT, color="#DD8452", linewidth=0.8, linestyle=":")
ax.annotate(f"floor = {CELL_FLOOR_DEFAULT}", xy=(CELL_FLOOR_DEFAULT, 0.02),
            xytext=(CELL_FLOOR_DEFAULT + 0.05, 0.10), fontsize=9, color="#DD8452")
ax.annotate(f"floor = {CELL_FLOOR_ROBUST_2}", xy=(CELL_FLOOR_ROBUST_2, 0.02),
            xytext=(CELL_FLOOR_ROBUST_2 + 0.05, 0.18), fontsize=9, color="#4C72B0")
ax.set_xlabel("hm_cell (ppb)")
ax.set_ylabel("Fractional size reduction")
ax.set_title("hm_cell mitigation")
ax.set_ylim(-0.02, 0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("hm_cell_mitigation.png", dpi=150)
print("\nSaved plot to hm_cell_mitigation.png")


# =============================================================================
# ROI mitigation
# =============================================================================

roi_fractions = np.linspace(0.0, 1.0, 300)
mitigation_values = [mitigation_roi(f, 1.0) for f in roi_fractions]

sample_fractions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
print(f"\n{'ROI fraction':>14} {'mitigation':>12}")
for f in sample_fractions:
    print(f"{f:>14.2f} {mitigation_roi(f, 1.0):>12.3f}")

fig, ax = plt.subplots(figsize=(5, 5/1.4))
ax.plot(roi_fractions, mitigation_values, label="ROI mitigation", color="#4C72B0")
ax.set_xlabel("ROI fraction (n_roi / n_chan)")
ax.set_ylabel("Fractional size reduction")
ax.set_title("ROI mitigation")
ax.set_ylim(-0.02, 1.02)
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("roi_mitigation.png", dpi=150)
print("\nSaved plot to roi_mitigation.png")


# =============================================================================
# nfields mitigation
# =============================================================================

# n_fields_kept ranges from 1 (only the representative field survives)
# up to n_fields_total (nothing dropped), for a few representative
# n_fields_total values.
n_fields_total_values = [2, 3, 5, 10]

fig, ax = plt.subplots(figsize=(5, 5/1.4))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for n_total, color in zip(n_fields_total_values, colors):
    n_kept_values = np.arange(1, n_total + 1)
    mitigation_kept = [mitigation_fields(n_total, n_kept) for n_kept in n_kept_values]
    ax.plot(n_kept_values, mitigation_kept, marker="o",
            label=f"n_fields_total = {n_total}", color=color)

sample_cases = [(5, 1), (5, 3), (5, 5), (10, 1)]
print(f"\n{'n_fields_total':>15} {'n_fields_kept':>14} {'mitigation':>12}")
for n_total, n_kept in sample_cases:
    print(f"{n_total:>15} {n_kept:>14} {mitigation_fields(n_total, n_kept):>12.3f}")

ax.set_xlabel("n_fields_kept")
ax.set_ylabel("Fractional size reduction")
ax.set_title("nfields mitigation")
ax.set_ylim(-0.02, 1.02)
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("nfields_mitigation.png", dpi=150)
print("\nSaved plot to nfields_mitigation.png")
