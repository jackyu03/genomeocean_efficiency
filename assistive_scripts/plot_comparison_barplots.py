import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for seaborn
sns.set_theme(style="whitegrid")

BASE_DIR = "/Users/JackYu_1/Desktop/genomeocean/genomeocean_efficiency/results/new"
FIGURES_DIR = "/Users/JackYu_1/Desktop/genomeocean/genomeocean_efficiency/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

###################################################################
# 1. Parse Protocol A
###################################################################
protocol_a_dir = os.path.join(BASE_DIR, "protocol_A")

a_perf_data = []
a_binning_data = []
a_quality_data = []

for folder in os.listdir(protocol_a_dir):
    if not folder.startswith("protocol_A_"): continue
    parts = folder.split("_")
    size = parts[2]
    seed = parts[4]
    
    folder_path = os.path.join(protocol_a_dir, folder)
    
    perf_file = os.path.join(folder_path, "performance_metrics.csv")
    if os.path.exists(perf_file):
        df = pd.read_csv(perf_file)
        df['size'] = size
        df['seed'] = seed
        a_perf_data.append(df)
        
    binning_file = os.path.join(folder_path, "binning_metrics.csv")
    if os.path.exists(binning_file):
        df = pd.read_csv(binning_file)
        df['size'] = size
        df['seed'] = seed
        a_binning_data.append(df)
        
    quality_file = os.path.join(folder_path, "quality", "quality_metrics.csv")
    if os.path.exists(quality_file):
        df = pd.read_csv(quality_file)
        df['size'] = size
        df['seed'] = seed
        a_quality_data.append(df)

df_a_perf = pd.concat(a_perf_data, ignore_index=True)
df_a_binning = pd.concat(a_binning_data, ignore_index=True)
df_a_quality = pd.concat(a_quality_data, ignore_index=True)

###################################################################
# 2. Parse Protocol B
###################################################################
protocol_b_dir = os.path.join(BASE_DIR, "protocol_B")

b_eff_data = []
b_qual_data = []

for folder in os.listdir(protocol_b_dir):
    if not folder.startswith("generation_bench_"): continue
    parts = folder.split("_")
    config = parts[2]
    seed = parts[4]
    
    folder_path = os.path.join(protocol_b_dir, folder)
    
    eff_file = os.path.join(folder_path, "generation_efficiency_metrics.csv")
    if os.path.exists(eff_file):
        df = pd.read_csv(eff_file)
        df['config'] = config
        df['seed'] = seed
        b_eff_data.append(df)
        
    qual_file = os.path.join(folder_path, "generation_quality_metrics.csv")
    if os.path.exists(qual_file):
        df = pd.read_csv(qual_file)
        df['config'] = config
        df['seed'] = seed
        mean_ppl = df['perplexity'].mean()
        b_qual_data.append({'config': config, 'seed': seed, 'mean_perplexity': mean_ppl})

df_b_eff = pd.concat(b_eff_data, ignore_index=True)
df_b_qual = pd.DataFrame(b_qual_data)

###################################################################
# 3. Print Aggregate Summaries for LaTeX Tables
###################################################################

print("="*60)
print("TABLE 3: PROTOCOL A AGGREGATE RESULTS")
print("="*60)

merged_a = pd.merge(
    df_a_perf[['size', 'quantization', 'seed', 'tokens_per_s', 'peak_vram_GB']],
    df_a_binning[['size', 'quantization', 'seed', 'ari']],
    on=['size', 'quantization', 'seed']
)

size_order = {'100M': 1, '500M': 2, '4B': 3}
merged_a['size_order'] = merged_a['size'].map(size_order)
agg_a = merged_a.groupby(['size_order', 'size', 'quantization']).agg({
    'tokens_per_s': ['mean', 'std'],
    'peak_vram_GB': ['mean', 'std'],
    'ari': ['mean', 'std']
}).reset_index()

for _, row in agg_a.iterrows():
    sz = row['size'].values[0]
    q = row['quantization'].values[0]
    tps_m = row[('tokens_per_s', 'mean')]
    tps_s = row[('tokens_per_s', 'std')]
    vram_m = row[('peak_vram_GB', 'mean')]
    vram_s = row[('peak_vram_GB', 'std')]
    ari_m = row[('ari', 'mean')]
    ari_s = row[('ari', 'std')]
    print(f"{sz} | {q.upper():<5} | Throughput: {tps_m:,.0f} +/- {tps_s:,.0f} | Peak VRAM: {vram_m:.2f} +/- {vram_s:.2f} | ARI: {ari_m:.4f} +/- {ari_s:.4f}")

print("\n"+"="*60)
print("TABLE 4: PROTOCOL B AGGREGATE RESULTS")
print("="*60)

merged_b = pd.merge(
    df_b_eff[['config', 'quantization', 'seed', 'throughput_tps', 'avg_power_W', 'tokens_per_watt']],
    df_b_qual,
    on=['config', 'seed']
)

config_order = {'baseline': 1, 'b1': 2, 'b2': 3}
merged_b['config_order'] = merged_b['config'].map(config_order)
agg_b = merged_b.groupby(['config_order', 'config']).agg({
    'throughput_tps': ['mean', 'std'],
    'avg_power_W': ['mean', 'std'],
    'tokens_per_watt': ['mean', 'std'],
    'mean_perplexity': ['mean', 'std']
}).reset_index()

for _, row in agg_b.iterrows():
    c = row['config'].values[0]
    tps_m = row[('throughput_tps', 'mean')]
    tps_s = row[('throughput_tps', 'std')]
    pow_m = row[('avg_power_W', 'mean')]
    pow_s = row[('avg_power_W', 'std')]
    eff_m = row[('tokens_per_watt', 'mean')]
    eff_s = row[('tokens_per_watt', 'std')]
    ppl_m = row[('mean_perplexity', 'mean')]
    ppl_s = row[('mean_perplexity', 'std')]
    print(f"{c:<10} | TPS: {tps_m:.2f} +/- {tps_s:.2f} | Power: {pow_m:.2f} +/- {pow_s:.2f} | Eff: {eff_m:.2f} +/- {eff_s:.2f} | PPL: {ppl_m:.2f} +/- {ppl_s:.2f}")


# Figure 3: DBSCAN metrics comparing BF16 and FP8 for 100M, 500M, 4B
df_a_binning_melted = df_a_binning.melt(
    id_vars=['size', 'quantization', 'seed'],
    value_vars=['ari', 'v_measure', 'homogeneity', 'completeness', 'silhouette'],
    var_name='Metric', value_name='Score'
)
df_a_binning_melted['Metric'] = df_a_binning_melted['Metric'].map({
    'ari': 'ARI', 'v_measure': 'V-Measure', 'homogeneity': 'Homogeneity',
    'completeness': 'Completeness', 'silhouette': 'Silhouette Score'
})

df_a_binning_melted['Condition'] = df_a_binning_melted['size'] + " " + df_a_binning_melted['quantization'].str.upper()
conditions_order = ['100M BF16', '100M FP8', '500M BF16', '500M FP8', '4B BF16', '4B FP8']
df_a_binning_melted['Condition'] = pd.Categorical(df_a_binning_melted['Condition'], categories=conditions_order, ordered=True)

plt.figure(figsize=(10, 9.6))
custom_palette = ['#a1c9f4', '#4c72b0', '#8de5a1', '#55a868', '#ff9f9b', '#c44e52']
ax = sns.barplot(
    data=df_a_binning_melted, 
    y="Metric", 
    x="Score", 
    hue="Condition",
    palette=custom_palette,
    errorbar="sd",
    orient="h"
)
sns.despine()
ax.set_xlabel("Score")
ax.set_ylabel("")
ax.set_title("DBSCAN Clustering Fidelity: BF16 vs. FP8 Across Model Scales", fontweight='bold')

# Put annotations on the root of each bar, in white
for p in ax.patches:
    w = p.get_width()
    # Check if w is valid (some patches might be empty)
    if not np.isnan(w) and w > 0:
        y = p.get_y() + p.get_height() / 2
        ax.text(0.01, y, f'{w:.2f}', color='white', ha='left', va='center', size=9, fontweight='bold')

plt.legend(title="Model & Precision", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_clustering_metrics_updated.pdf'), format='pdf')
plt.close('all')

# Figure 4: FP8 embedding fidelity metrics relative to BF16
df_a_quality['size'] = pd.Categorical(df_a_quality['size'], categories=['100M', '500M', '4B'], ordered=True)
df_a_quality['layer'] = df_a_quality['layer'].map({'last': 'Last Layer', 'second_last': 'Second to Last Layer'})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle('FP8 Embedding Fidelity: BF16 Reference vs. FP8 Quantized', y=0.88, fontsize=14, fontweight='bold')

sns.barplot(data=df_a_quality, x="size", y="cosine_similarity", hue="layer", ax=axes[0], errorbar="sd", palette="deep")
axes[0].set_title('Cosine Similarity (Higher is better)')
axes[0].set_ylim(0.99, 1.0)
axes[0].set_ylabel('Score')

sns.barplot(data=df_a_quality, x="size", y="snr_db", hue="layer", ax=axes[1], errorbar="sd", palette="deep")
axes[1].set_title('Signal-to-Noise Ratio (dB, Higher is better)')
axes[1].set_ylabel('SNR (dB)')

sns.barplot(data=df_a_quality, x="size", y="kl_divergence", hue="layer", ax=axes[2], errorbar="sd", palette="deep")
axes[2].set_title('KL Divergence (Lower is better)')
axes[2].set_ylabel('KL Divergence')

for ax in axes:
    ax.set_xlabel('Model Size')

# Annotations on the tip of each bar, in white, except for 4B's KL divergence.
for i, ax in enumerate(axes):
    fmt = '.4f' if i == 0 or i == 2 else '.1f'
    ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    # seaborn barplot orders patches by hue then x.
    # We have 2 hues, 3 sizes = 6 patches. 
    # patches[2] is hue0/size2 (4B), patches[5] is hue1/size2 (4B).
    for j, p in enumerate(ax.patches):
        h = p.get_height()
        if not np.isnan(h):
            x = p.get_x() + p.get_width() / 2
            if i == 2 and j in [2, 5]: # 4B for KL divergence
                y_pos = h + (ylim_range * 0.05)
                ax.text(x, y_pos, f'{h:{fmt}}', color='black', ha='center', va='bottom', size=9, fontweight='bold')
            else:
                y_pos = h - (ylim_range * 0.08)
                ax.text(x, y_pos, f'{h:{fmt}}', color='white', ha='center', va='top', size=9, fontweight='bold')

handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    ax.get_legend().remove()
# Legend positioning
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.83), ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.72)
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_fidelity_metrics_updated.pdf'), format='pdf')
plt.close('all')

# Figure 5: Throughput comparison of BF16 and FP8 inference
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle('FP8 vs. BF16 Throughput Across Experiment Protocols', y=0.98, fontsize=14, fontweight='bold')

# Panel 1: Protocol A
df_a_perf['size'] = pd.Categorical(df_a_perf['size'], categories=['100M', '500M', '4B'], ordered=True)
sns.barplot(data=df_a_perf, x="size", y="tokens_per_s", hue="quantization", ax=axes[0], errorbar="sd", palette="muted")
axes[0].set_yscale("log")
axes[0].set_title("Protocol A: Embedding Extraction")
axes[0].set_xlabel("Model Size")
axes[0].set_ylabel("Throughput (tokens / second)")

means_a = df_a_perf.groupby(['size', 'quantization'])['tokens_per_s'].mean().unstack()
for i, size in enumerate(['100M', '500M', '4B']):
    val_bf16 = means_a.loc[size, 'bf16']
    val_fp8 = means_a.loc[size, 'fp8']
    pct_change = (val_fp8 - val_bf16) / val_bf16 * 100
    
    bar = axes[0].containers[1][i]
    height = bar.get_height()
    y_pos = height * 1.2
    axes[0].text(bar.get_x() + bar.get_width()/2., y_pos, f"{pct_change:+.1f}%", ha='center', va='bottom', color='black', fontweight='bold')

# Panel 2: Protocol B
merged_b['config'] = pd.Categorical(merged_b['config'], categories=['baseline', 'b1', 'b2'], ordered=True)
sns.barplot(data=merged_b, x="config", y="throughput_tps", ax=axes[1], errorbar="sd", palette="muted")
axes[1].set_title("Protocol B: Autoregressive Generation (4B)")
axes[1].set_xlabel("Quantization Configuration")
axes[1].set_ylabel("Throughput (tokens / second)")
axes[1].set_xticklabels(['Baseline (BF16)', 'B.1 (FP8 KV Cache)', 'B.2 (W8A8 Dynamic)'])

means_b = merged_b.groupby('config')['throughput_tps'].mean()
base_val = means_b['baseline']
bars = axes[1].patches
for i, conf in enumerate(['b1', 'b2']):
    val = means_b[conf]
    pct_change = (val - base_val) / base_val * 100
    
    bar = bars[i+1]
    height = bar.get_height()
    y_pos = height + (axes[1].get_ylim()[1] * 0.02)
    axes[1].text(bar.get_x() + bar.get_width()/2., y_pos, f"{pct_change:+.1f}%", ha='center', va='bottom', color='black', fontweight='bold')

axes[1].axhline(y=base_val * 2, color='gray', linestyle='--', label='Theoretical 2x (Dense Matrix FP8)')
axes[1].legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_throughput_updated.pdf'), format='pdf')
plt.close('all')

print("\nPlots generated and saved to ./figures as .pdf")
