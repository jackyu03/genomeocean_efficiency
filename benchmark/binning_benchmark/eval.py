"""
Binning Evaluation Logic

Implements DBSCAN clustering and metrics calculation for binning evaluation.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.preprocessing import normalize
from collections import Counter

log = logging.getLogger(__name__)

def run_binning_eval(embeddings: np.ndarray, 
                    genome_ids: List[str], 
                    method: str = "dbscan",
                    **kwargs) -> Dict[str, float]:
    """
    Runs clustering on embeddings and computes metrics against ground truth genome_ids.
    
    Args:
        embeddings: (N, D) array of embeddings
        genome_ids: List of ground truth labels of length N
        method: 'dbscan' or 'kmeans'
        **kwargs: Clustering parameters (eps, min_samples, n_clusters)
        
    Returns:
        Dictionary of metrics
    """
    # Normalize for cosine-like distance in Euclidean space
    embeddings_norm = normalize(embeddings)
    
    labels_pred = -1 * np.ones(len(embeddings))
    
    if method == "dbscan":
        eps = kwargs.get("eps", 0.05)
        min_samples = kwargs.get("min_samples", 3)
        log.info(f"Clustering with DBSCAN(eps={eps}, min_samples={min_samples})...")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        labels_pred = clusterer.fit_predict(embeddings_norm)
        
    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 10)
        log.info(f"Clustering with K-Means(k={n_clusters})...")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_pred = clusterer.fit_predict(embeddings_norm)
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # Metrics
    # Map string labels to int
    unique_genomes = list(set(genome_ids))
    genome_map = {g: i for i, g in enumerate(unique_genomes)}
    labels_true = np.array([genome_map[g] for g in genome_ids])
    
    ari = adjusted_rand_score(labels_true, labels_pred)
    hom = homogeneity_score(labels_true, labels_pred)
    comp = completeness_score(labels_true, labels_pred)
    v_meas = v_measure_score(labels_true, labels_pred)
    
    # Custom Purity (excluding noise)
    unique_bins = set(labels_pred)
    if -1 in unique_bins:
        unique_bins.remove(-1)
        
    n_bins = len(unique_bins)
    n_noise = np.sum(labels_pred == -1)
    contamination_rate = n_noise / len(labels_pred)
    
    purities = []
    if n_bins > 0:
        for b in unique_bins:
            mask = (labels_pred == b)
            bin_true_labels = [genome_ids[i] for i in np.where(mask)[0]]
            # Purity = max count of single species / total in bin
            counts = Counter(bin_true_labels)
            most_common_count = counts.most_common(1)[0][1]
            purities.append(most_common_count / len(bin_true_labels))
        avg_purity = np.mean(purities)
    else:
        avg_purity = 0.0
        
    log.info(f"Binning Results: ARI={ari:.3f}, AvgPurity={avg_purity:.3f}, Bins={n_bins}, Noise={n_noise}")
    
    return {
        "ari": ari,
        "homogeneity": hom,
        "completeness": comp,
        "v_measure": v_meas,
        "avg_purity": avg_purity,
        "n_bins_formed": n_bins,
        "fraction_noise": contamination_rate
    }
