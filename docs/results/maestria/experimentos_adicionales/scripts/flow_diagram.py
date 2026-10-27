import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(16.4, 9.7))
ax.set_xlim(0, 17.0)
ax.set_ylim(0.8, 15.1)
ax.axis("off")

COL_A_CX = 2.2    # RSNA training branch
COL_B_CX = 6.7    # RSNA validation branch
COL_C_CX = 11.0   # RSNA official test split (unused, dead end)
COL_D_CX = 14.8   # Mexican cohort branch (fully independent, own column)

FS = 13.5


def box(cx, y, w, h, text, fc="#eaf1fb", ec="#3b6ea5", fontsize=FS):
    x = cx - w / 2
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.09",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.6)
    return dict(top=(cx, y + h), bottom=(cx, y), left=(x, y + h / 2), right=(x + w, y + h / 2),
                cx=cx, y=y, h=h)


def vline(p1, p2, color="#444", lw=1.5):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color,
                         connectionstyle="arc3,rad=0.0", shrinkA=0, shrinkB=2)
    ax.add_patch(a)


ax.text(8.4, 14.7, "Participant Flow Diagram", ha="center", fontsize=18, weight="bold")

# ---- Row 1: RSNA source (spans columns A+B+C) ----
n1 = box(6.7, 13.1, 10.7, 1.25, "RSNA Pediatric Bone Age Challenge 2017\n14,236 radiographs")
vline((COL_A_CX, n1["y"]), (COL_A_CX, 12.35))
vline((COL_B_CX, n1["y"]), (COL_B_CX, 12.35))
vline((COL_C_CX, n1["y"]), (COL_C_CX, 12.35))

# ---- Row 1 (col D): Mexican Clinical Cohort, fully independent source, own column ----
n_mex1 = box(COL_D_CX, 13.1, 3.6, 1.25, "Mexican Clinical Dataset\n100 radiographs",
             fc="#eafbea", ec="#2e8b57")

# ---- Row 2: three-way official split ----
n2a = box(COL_A_CX, 10.65, 4.0, 1.5, "Official Training Set\n12,611 images")
n2b = box(COL_B_CX, 10.65, 4.0, 1.5, "Official Validation Set\n1425 images (held out)")
n2c = box(COL_C_CX, 10.65, 3.6, 1.5, "Official Test Set\n200 images, not used",
          fc="#fbeaea", ec="#c0392b")

# ---- Row 3: age filter (A) ----
n3 = box(COL_A_CX, 8.5, 4.0, 1.3, "Age-Frequency Filter\n(≥ 50 images/month)\n−828 images excluded")
vline(n2a["bottom"], n3["top"])

# ---- Row 4: balanced dataset (A) ----
n4 = box(COL_A_CX, 6.35, 4.0, 1.4, "Balanced Subset\n11,783 images")
vline(n3["bottom"], n4["top"])

# ---- Row 5: internal train / validation split (A only, terminal for this branch) ----
n5a = box(COL_A_CX - 1.05, 4.25, 1.9, 1.4, "Internal\nTraining Subset\n9426")
n5b = box(COL_A_CX + 1.05, 4.25, 1.9, 1.4, "Internal\nValidation Subset\n2357")
vline(n4["bottom"], (n5a["cx"], n5a["y"] + n5a["h"]))
vline(n4["bottom"], (n5b["cx"], n5b["y"] + n5b["h"]))

# ---- Row 6: external evaluations (B and D), each fed directly by its held-out data ----
n4b = box(COL_B_CX, 1.0, 4.0, 1.5, "Independent RSNA\nValidation",
          fc="#fff6e0", ec="#b8860b")
vline(n2b["bottom"], n4b["top"])

n_mex2 = box(COL_D_CX, 1.0, 3.6, 1.5, "External Mexican\nValidation",
             fc="#eafbea", ec="#2e8b57")
vline(n_mex1["bottom"], n_mex2["top"])

plt.tight_layout()
fig.savefig("/lustre/home/mlozano/boneage-predictor/review2/figures/participant_flow_diagram.png",
            dpi=150, bbox_inches="tight")
print("Diagram saved.")
