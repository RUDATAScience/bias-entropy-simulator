import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
from google.colab import files

# ==========================================
# [Setup] Create output directory
# ==========================================
output_dir = "rigorous_validation_results"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Core Model Definition ---
def u_base(i, peak):
    return 1.0 - 0.25 * np.abs(i - peak)

def softmax(utilities, beta):
    exp_u = np.exp(beta * utilities)
    return exp_u / np.sum(exp_u)

def calculate_dist(v2, beta):
    options = np.array([1, 2, 3, 4, 5])
    U1 = (1 - v2) * u_base(options, 1) + v2 * u_base(options, 4)
    U2 = (1 - v2) * u_base(options, 3) + v2 * u_base(options, 4)
    return 0.10 * softmax(U1, beta) + 0.90 * softmax(U2, beta)

def shannon_entropy(p):
    return -np.sum(p * np.log2(p + 1e-12))

# --- 2. Validation A: Heatmap Sensitivity Analysis (Phase Diagram) ---
print("Running Validation A: Parameter Sensitivity Heatmap...")
v2_space = np.linspace(0, 1, 50)
beta_space = np.linspace(1, 10, 50)
heatmap_data = np.zeros((len(beta_space), len(v2_space)))

for i, b in enumerate(beta_space):
    for j, v in enumerate(v2_space):
        dist = calculate_dist(v, b)
        heatmap_data[i, j] = dist[0]  # Probability of Rating 1

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, xticklabels=np.round(v2_space, 2), yticklabels=np.round(beta_space, 1),
            cmap="YlOrRd_r", cbar_kws={'label': 'Prob(Rating 1)'})
plt.title("Fig A: Phase Diagram of Signal Evaporation (Beta vs v2)", fontsize=14)
plt.xlabel("Sontaku Weight (v2)", fontsize=12)
plt.ylabel("Certainty (Beta)", fontsize=12)
plt.xticks(np.arange(0, 50, 5), np.round(v2_space[::5], 2))
plt.yticks(np.arange(0, 50, 5), np.round(beta_space[::5], 1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_A_phase_diagram.png'), dpi=300)
plt.close()

# --- 3. Validation B: Informational Entropy Decay ---
print("Running Validation B: Entropy Decay Analysis...")
v2_range = np.linspace(0, 1, 100)
entropy_list = [shannon_entropy(calculate_dist(v, 5.0)) for v in v2_range]

plt.figure(figsize=(8, 5))
plt.plot(v2_range, entropy_list, color='green', linewidth=3)
plt.title("Fig B: Information Entropy Decay (Systemic Homogenization)", fontsize=14)
plt.xlabel("Sontaku Weight (v2)", fontsize=12)
plt.ylabel("Shannon Entropy (bits)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_B_entropy_decay.png'), dpi=300)
plt.close()

# --- 4. Validation C: The Pathology of Outlier Removal ---
print("Running Validation C: Outlier Removal Impact...")
N_test = 20000
v2_targets = [0.0, 0.25, 0.5, 0.75]
options = np.array([1, 2, 3, 4, 5])
filtering_results = []

true_mean_baseline = np.sum(options * calculate_dist(0.0, 5.0))

for v in v2_targets:
    dist = calculate_dist(v, 5.0)
    # Generate population
    data = np.random.choice(options, size=N_test, p=dist)
    raw_mean = np.mean(data)
    
    # 3-Sigma Outlier Removal (Simplified: Remove Rating 1 if it's too far)
    mean = np.mean(data)
    std = np.std(data)
    filtered_data = data[np.abs(data - mean) <= 2 * std] # Using 2-sigma to be strict
    filtered_mean = np.mean(filtered_data)
    
    filtering_results.append({
        "v2": v,
        "Raw_Mean": raw_mean,
        "Filtered_Mean": filtered_mean,
        "Bias_Increase_Percent": (np.abs(filtered_mean - true_mean_baseline) - np.abs(raw_mean - true_mean_baseline)) / np.abs(raw_mean - true_mean_baseline) * 100
    })

df_filter = pd.DataFrame(filtering_results)
df_filter.to_csv(os.path.join(output_dir, 'data_C_filtering_pathology.csv'), index=False)

# Graph C: Gap between Filtered and True Mean
plt.figure(figsize=(8, 6))
plt.plot(df_filter["v2"], df_filter["Raw_Mean"], marker='o', label='Observed Mean (Raw)')
plt.plot(df_filter["v2"], df_filter["Filtered_Mean"], marker='s', label='Observed Mean (Filtered/Cleaned)', linestyle='--')
plt.axhline(y=true_mean_baseline, color='blue', linestyle=':', label='True Mean (Ground Truth)')
plt.title("Fig C: How 'Data Cleaning' Amplifies Bias", fontsize=14)
plt.xlabel("Sontaku Weight (v2)", fontsize=12)
plt.ylabel("Mean Value", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig_C_filtering_bias.png'), dpi=300)
plt.close()

# --- 5. Zip and Download ---
print("Compressing results...")
zip_filename = "rigorous_validation_archive"
shutil.make_archive(zip_filename, 'zip', output_dir)
files.download(f"{zip_filename}.zip")
print("✅ Done. Critical validation archive is ready.")
