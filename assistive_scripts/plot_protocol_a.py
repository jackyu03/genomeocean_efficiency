import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import urllib.request
from urllib.error import HTTPError
import json
import ssl
import time

# Hardcoded dictionary based on GTDB NCBI Entrez API query
# This provides biological strain-level names without requiring an internet connection.
GTDB_TO_SPECIES = {
    "GCA_018831195.1": "Candidatus Eiseniibacteriota bacterium",
    "GCF_021729845.1": "Rhizobium sp. SL42 str. SL42",
    "GCF_900144675.1": "Pseudomonas aeruginosa",
    "GCA_000170555.1": "Burkholderia pseudomallei str. 112",
    "GCF_000293135.1": "Klebsiella sp. OBRC7 str. OBRC7",
    "GCF_027533845.1": "Streptomyces sp. RPT161 str. RPT161",
    "GCF_016622475.1": "Pseudomonas aeruginosa str. KOL14.W.495.33",
    "GCF_002227195.1": "Escherichia coli str. MOD1-EC5662",
    "GCF_000403665.1": "Streptomyces lividans str. 1326",
    "GCF_000619145.1": "Escherichia coli str. 2009EL1449",
    "GCF_014076515.1": "Pseudomonas putida str. T25-27",
    "GCF_002531305.1": "Escherichia coli str. OLC1356",
    "GCF_004369915.1": "Pseudomonas aeruginosa str. 22.1",
    "GCA_937868145.1": "uncultured Phycisphaerales bacterium",
    "GCF_900015995.1": "Escherichia coli str. M858",
    "GCF_005398985.1": "Escherichia coli str. 04-A81-A",
    "GCA_900451505.1": "Klebsiella pneumoniae str. NCTC13810",
    "GCF_006374815.1": "Vibrio parahaemolyticus str. B9_5",
    "GCA_025963165.1": "Mycobacterium sp.",
    "GCF_030016275.1": "Streptomyces coelicoflavus str. S3018",
}

def fetch_with_retry(acc):
    """Fallback NCBI Entrez lookup with retries for accessions not in the dictionary."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    retries = 3
    for attempt in range(retries):
        try:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=assembly&term={acc}&retmode=json"
            with urllib.request.urlopen(search_url, context=ctx) as response:
                data = json.loads(response.read().decode())
                
            if not data["esearchresult"]["idlist"]:
                acc_base = acc.split('.')[0]
                search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=assembly&term={acc_base}&retmode=json"
                with urllib.request.urlopen(search_url, context=ctx) as response:
                    data = json.loads(response.read().decode())
                    
            if not data["esearchresult"]["idlist"]:
                return acc
                
            uid = data["esearchresult"]["idlist"][0]
            time.sleep(0.5)
            
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={uid}&retmode=json"
            with urllib.request.urlopen(summary_url, context=ctx) as response:
                data2 = json.loads(response.read().decode())
                
            species = data2["result"][uid].get("speciesname", "")
            strain = ""
            biosource = data2["result"][uid].get("biosource", {})
            for infra in biosource.get("infraspecieslist", []):
                if infra.get("sub_type") == "strain":
                    strain = infra.get("sub_value", "")
                    break
            
            if species and strain:
                return f"{species} str. {strain}"
            return species if species else acc

        except HTTPError as e:
            if e.code == 429:
                print(f"  [429] Rate limited on {acc}, waiting 5 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(5)
            else:
                return acc
        except Exception as e:
            print(f"  Error on {acc}: {e}")
            return acc
            
    return acc

def plot_protocol_a_grid(dir_100m, dir_4b, output_path):
    """
    Plots a 2x2 grid showing Ground Truth species clusters:
    Rows: BF16, FP8
    Columns: 100M, 4B
    Legend: Vertical color band on the far right with biological strain names.
    """
    dirs = {
        ("100M", "bf16"): os.path.join(dir_100m, "plots", "bf16", "cluster_data.npz"),
        ("100M", "fp8"):  os.path.join(dir_100m, "plots", "fp8", "cluster_data.npz"),
        ("4B", "bf16"):   os.path.join(dir_4b, "plots", "bf16", "cluster_data.npz"),
        ("4B", "fp8"):    os.path.join(dir_4b, "plots", "fp8", "cluster_data.npz"),
    }
    
    data = {}
    all_accessions = set()
    for key, path in dirs.items():
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            data[key] = {
                'emb_2d': d['emb_2d'],
                'genome_ids': d['genome_ids']
            }
            all_accessions.update(d['genome_ids'])
        else:
            print(f"Error: {path} not found.")
            print(f"Please ensure both 100M and 4B models have been run with run_embedding_benchmark.py and include the 'plots' folder.")
            return

    # Use the hardcoded dictionary or fallback
    name_mapping = {}
    for acc in sorted(list(all_accessions)):
        if acc in GTDB_TO_SPECIES:
            name_mapping[acc] = GTDB_TO_SPECIES[acc]
        else:
            print(f"Accession {acc} not in hardcoded dictionary. Fetching from NCBI...")
            name_mapping[acc] = fetch_with_retry(acc)
            # Sleep 1.5s to respect rate limits if we have to fetch multiple
            time.sleep(1.5)

    n_species = len(all_accessions)
    palette = sns.color_palette("tab20", n_species)
    color_map = {acc: palette[i] for i, acc in enumerate(sorted(list(all_accessions)))}
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.3], wspace=0.1, hspace=0.1)
    
    ax_100_bf16 = fig.add_subplot(gs[0, 0])
    ax_4b_bf16  = fig.add_subplot(gs[0, 1])
    ax_100_fp8  = fig.add_subplot(gs[1, 0])
    ax_4b_fp8   = fig.add_subplot(gs[1, 1])
    
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis('off')
    
    def draw_scatter(ax, key):
        emb = data[key]['emb_2d']
        labels = data[key]['genome_ids']
        
        for acc in sorted(list(all_accessions)):
            mask = np.array(labels) == acc
            if np.any(mask):
                ax.scatter(emb[mask, 0], emb[mask, 1], c=[color_map[acc]], s=10, alpha=0.8, edgecolors='none')
                
        ax.set_xticks([])
        ax.set_yticks([])
        
    draw_scatter(ax_100_bf16, ("100M", "bf16"))
    draw_scatter(ax_4b_bf16,  ("4B", "bf16"))
    draw_scatter(ax_100_fp8,  ("100M", "fp8"))
    draw_scatter(ax_4b_fp8,   ("4B", "fp8"))
    
    # Add main plot titles
    ax_100_bf16.set_title("GenomeOcean-100M", fontsize=14, fontweight='bold', pad=10)
    ax_4b_bf16.set_title("GenomeOcean-4B", fontsize=14, fontweight='bold', pad=10)
    
    ax_100_bf16.set_ylabel("BF16", fontsize=14, fontweight='bold', labelpad=10)
    ax_100_fp8.set_ylabel("FP8", fontsize=14, fontweight='bold', labelpad=10)
    
    # Draw Legend
    y_starts = np.linspace(1, 0, n_species + 1)
    height = y_starts[0] - y_starts[1]
    
    for i, acc in enumerate(sorted(list(all_accessions))):
        color = color_map[acc]
        name = name_mapping[acc]
        y_pos = y_starts[i+1]
        
        rect = mpatches.Rectangle((0, y_pos), 0.1, height, facecolor=color, edgecolor='white', linewidth=1)
        ax_legend.add_patch(rect)
        ax_legend.text(0.15, y_pos + height/2, name, va='center', ha='left', fontsize=9)
        
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Protocol A Unified Grid successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master plotting script for Protocol A (100M vs 4B grid)")
    parser.add_argument("--100m-dir", type=str, required=True, help="Path to the 100M output folder (e.g., results/new/protocol_A_100M_seed_42)")
    parser.add_argument("--4b-dir", type=str, required=True, help="Path to the 4B output folder (e.g., results/new/protocol_A_4B_seed_42)")
    parser.add_argument("--output", type=str, default="figures/genomeocean_protocol_A_grid.png")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_protocol_a_grid(getattr(args, '100m_dir'), getattr(args, '4b_dir'), args.output)
