"""
Binning Evaluation Logic

Implements UMAP reduction, DBSCAN clustering, Visualization, and Metrics.
"""
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from collections import Counter

# Try importing UMAP, handle if missing
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

log = logging.getLogger(__name__)

def plot_clusters(embeddings_2d: np.ndarray, 
                 labels: List, 
                 title: str, 
                 output_path: str):
    """
    Generates a scatter plot of the 2D embeddings.
    """
    plt.figure(figsize=(10, 8))
    
    # Use seaborn for better handling of categorical labels/colors
    # Convert labels to string if they are numbers to ensure discrete palette
    labels_str = [str(x) for x in labels]
    
    n_unique = len(set(labels))
    palette = sns.color_palette("tab10", n_unique) if n_unique <= 10 else "viridis"
    
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels_str,
        palette=palette,
        s=10,
        alpha=0.7,
        legend="full" if n_unique < 30 else False # Hide legend if too many classes
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_binning_eval(embeddings: np.ndarray, 
                    genome_ids: List[str], 
                    method: str = "dbscan",
                    output_dir: str = ".",
                    umap_dim: int = 10,
                    **kwargs) -> Dict[str, float]:
    """
    Full pipeline:
    1. UMAP Reduction 
       - Projects to `umap_dim` for Clustering
       - Projects to 2D separately for Visualization
    2. Clustering (DBSCAN)
    3. Metrics
    4. Visualization (Ground Truth vs Predicted)
    """
    
    clustering_input = embeddings
    emb_2d = None
    
    if UMAP_AVAILABLE:
        log.info(f"Running UMAP reduction (Target Dim={umap_dim})...")
        
        # 1a. Reduce for Clustering (e.g., 10D or 50D)
        # Using a higher dimension than 2 often helps DBSCAN separate nearby manifolds
        if umap_dim > 0 and embeddings.shape[1] > umap_dim:
            reducer_clus = umap.UMAP(n_components=umap_dim, random_state=42, n_neighbors=30, min_dist=0.0)
            clustering_input = reducer_clus.fit_transform(embeddings)
        
        # 1b. Reduce for Visualization (Strictly 2D)
        log.info("Running UMAP reduction (2D) for visualization...")
        reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        emb_2d = reducer_2d.fit_transform(embeddings)
        
    else:
        log.warning("UMAP not installed. Clustering on raw embeddings.")
        
    # 2. Clustering
    # Normalize scale for DBSCAN (Euclidean distance is sensitive to scale)
    clustering_input = StandardScaler().fit_transform(clustering_input) 
    
    log.info(f"Clustering with {method} on {clustering_input.shape[1]}-dim features...")
    if method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels_pred = clusterer.fit_predict(clustering_input)
    else:
        # K-means fallback
        clusterer = KMeans(n_clusters=kwargs.get("n_clusters", 50))
        labels_pred = clusterer.fit_predict(clustering_input)

    # 3. Metrics
    unique_genomes = list(set(genome_ids))
    genome_map = {g: i for i, g in enumerate(unique_genomes)}
    labels_true = np.array([genome_map[g] for g in genome_ids])
    
    ari = adjusted_rand_score(labels_true, labels_pred)
    v_meas = v_measure_score(labels_true, labels_pred)
    hom = homogeneity_score(labels_true, labels_pred)
    comp = completeness_score(labels_true, labels_pred)
    
    # Silhouette
    try:
        # Calculate silhouette on the CLUSTERING input space (e.g. 10D UMAP)
        if len(set(labels_pred)) > 1:
            sil = silhouette_score(clustering_input, labels_pred, sample_size=5000)
        else:
            sil = -1.0
    except:
        sil = -1.0
        
    # Purity
    unique_bins = set(labels_pred) - {-1}
    purities = []
    if unique_bins:
        for b in unique_bins:
            mask = (labels_pred == b)
            bin_true = [genome_ids[i] for i in np.where(mask)[0]]
            common, count = Counter(bin_true).most_common(1)[0]
            purities.append(count / len(bin_true))
        avg_purity = np.mean(purities)
    else:
        avg_purity = 0.0

    n_noise = np.sum(labels_pred == -1)
    
    # 4. Visualization (Always use the 2D projection)
    if emb_2d is not None and output_dir:
        import os
        # Plot Prediction
        plot_clusters(emb_2d, labels_pred, 
                     title=f"Predicted Clusters (DBSCAN)\nARI={ari:.2f}", 
                     output_path=os.path.join(output_dir, "cluster_viz_predicted.png"))
        
        # Plot Ground Truth
        plot_clusters(emb_2d, genome_ids, 
                     title="Ground Truth Species", 
                     output_path=os.path.join(output_dir, "cluster_viz_truth.png"))
        
    log.info(f"Metrics: ARI={ari:.3f}, Purity={avg_purity:.3f}, Sil={sil:.3f}, Noise={n_noise}")
    
    return {
        "ari": ari,
        "v_measure": v_meas,
        "homogeneity": hom,
        "completeness": comp,
        "silhouette": sil,
        "avg_purity": avg_purity,
        "n_clusters": len(unique_bins),
        "n_noise": n_noise
    }
