import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==== CONFIGURATION ==========================================================
INPUT_CSV = "validation_complete_metrics.csv"
NUM_LANDMARKS = 25
OUTPUT_PLOT = "landmark_boxplot.png"

# TOGGLE: Set to True to see every extreme miss (dots), False for a cleaner box view
SHOW_OUTLIERS = False 
# =============================================================================

# ==== DATA PROCESSING ========================================================
print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# We collect data image-by-image, landmark-by-landmark 
# to ensure we only include visible points (v=2)
plot_records = []
all_valid_nme_values = []

print("Filtering for visible landmarks (v=2)...")
for i in range(1, NUM_LANDMARKS + 1):
    v_col = f'L{i}_gt_v'
    nme_col = f'L{i}_nme'
    
    # Filter rows where this specific landmark is visible
    visible_subset = df[df[v_col] == 2][nme_col].dropna()
    
    for val in visible_subset:
        plot_records.append({'Landmark': i, 'NME': val})
        all_valid_nme_values.append(val)

# Create a long-form DataFrame for Seaborn
plot_df = pd.DataFrame(plot_records)

# Calculate Overall Mean from visible entries only
overall_mean_nme = np.mean(all_valid_nme_values) if all_valid_nme_values else 0

# ==== PLOTTING ===============================================================
plt.figure(figsize=(16/2, 8/2))

# Define outlier style (small grey dots if enabled)
flierprops = dict(marker='o', markerfacecolor='gray', markersize=3,
                  linestyle='none', markeredgecolor='none', alpha=0.4)

# 1. Create the Boxplot
# showfliers is controlled by the bool at the top
sns.boxplot(
    data=plot_df, 
    x='Landmark', 
    y='NME', 
    palette='viridis', 
    showfliers=SHOW_OUTLIERS,
    flierprops=flierprops if SHOW_OUTLIERS else None,
    width=0.7
)

# 2. Add Overall Mean Line (Visible only)
plt.axhline(
    y=overall_mean_nme, 
    color='red', 
    linestyle='--', 
    linewidth=2, 
    label=f'Global Mean NME: {overall_mean_nme:.5f}'
)

# Formatting
title_suffix = "(With Outliers)" if SHOW_OUTLIERS else "(Outliers Hidden)"
plt.title(f'NME Distribution per Landmark {title_suffix}', fontsize=16)
plt.xlabel('Landmark Index', fontsize=12)
plt.ylabel('Normalized Mean Error', fontsize=12)

# Set Y-limit
# If outliers are hidden, we zoom in. If shown, we let the scale expand to fit them.
if not SHOW_OUTLIERS and all_valid_nme_values:
    # plt.ylim(0, max(overall_mean_nme * 4, 0.1)) 
    pass
elif SHOW_OUTLIERS:
    # Cap the Y-axis at 1.0 (diagonal length) as NME rarely exceeds this 
    # and it keeps the plot from becoming infinitely tall due to noise
    plt.ylim(0, min(max(all_valid_nme_values) * 1.0, 1.0))

plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(axis='y', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"✅ Success! Plot saved as '{OUTPUT_PLOT}'")
print(f"📊 Global Mean NME for visible landmarks: {overall_mean_nme:.5f}")

plt.show()