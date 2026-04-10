"""
Binning Benchmark Utils

Utilities for preparing datasets for the binning task.
"""
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def prepare_binning_dataset(df: pd.DataFrame, 
                          n_species: int = 50, 
                          min_length: int = 1000, 
                          seed: int = 42) -> pd.DataFrame:
    """
    Selects a subset of the dataset for binning evaluation.
    
    Args:
        df: Input DataFrame with 'genome_id' and 'seq'
        n_species: Number of species to select
        min_length: Minimum sequence length to keep
        seed: Random seed
        
    Returns:
        Filtered DataFrame
    """
    rng = np.random.default_rng(seed)
    
    # Filter by length
    if "seq" in df.columns:
        df["seq_len"] = df["seq"].str.len()
        df = df[df["seq_len"] >= min_length].copy()
    
    available_species = df["genome_id"].unique()
    log.info(f"Preparing binning dataset. Found {len(available_species)} species with seqs >= {min_length}bp.")
    
    if len(available_species) < n_species:
        log.warning(f"Requested {n_species} species, but only {len(available_species)} available. Using all.")
        selected_species = available_species
    else:
        # Prioritize species with more fragments? Or just random?
        # Random for now to simulate general metagenome
        selected_species = rng.choice(available_species, size=n_species, replace=False)
        
    subset = df[df["genome_id"].isin(selected_species)].copy()
    
    log.info(f"Selected {len(selected_species)} species. Total fragments: {len(subset)}")
    return subset
