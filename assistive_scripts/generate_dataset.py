import os
import numpy as np
import gzip
import random
import pandas as pd
import threading
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def generate_genome_id_list(directory):
    np.random.seed(42)
    global labels
    
    # set operation to filter out only genome_ids present in the tree file (has phylo dist)
    selected_genome_ids = [filename.replace('.fna.gz', '') for filename in tqdm(os.listdir(directory)) if filename.endswith(".fna.gz")]
    
    # Sort the genome IDs to ensure deterministic order
    selected_genome_ids.sort()
    
    return selected_genome_ids

def process_genome(genome_id, directory, seqs_per_genome, seq_length, seed):
    """
    Process a single genome:
    1. Read and concat all sequences.
    2. Partition into non-overlapping chunks.
    3. Randomly select exactly `seqs_per_genome` chunks.
    4. If not enough chunks, FAIL.
    """
    try:
        file_path = os.path.join(directory, f"{genome_id}.fna.gz")
        if not os.path.exists(file_path):
            return [], False

        seqs = []
        with gzip.open(file_path, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                seqs.append(str(record.seq))

        full_seqs = "".join(seqs)
        
        # Partition into non-overlapping chunks of exact length
        all_chunks = [full_seqs[i : i + seq_length] for i in range(0, len(full_seqs), seq_length)]
        valid_chunks = [chunk for chunk in all_chunks if len(chunk) == seq_length]
        
        # Strict CHECK: Must have enough non-overlapping chunks
        if len(valid_chunks) < seqs_per_genome:
            return [], False
            
        # Deterministic sampling
        rng = random.Random(seed)
        selected_fragments = rng.sample(valid_chunks, seqs_per_genome)
        
        # Format: (genome_id, seq)
        # Using the unique fragment ID as the 'genome_id' column as requested
        data_rows = []
        for i, fragment in enumerate(selected_fragments):
            unique_id = f"{genome_id}_{i}"
            data_rows.append((unique_id, fragment))

        return data_rows, True

    except Exception:
        return [], False

def sample_sequences_parallel(directory, id_list, seqs_per_genome=100, seq_length=10000, n_species=50, seed=42):
    """
    Continuously attempts to sample genomes until `n_species` successful ones are found.
    """
    random.seed(seed)
    
    # Shuffle the ENTIRE ID list once deterministically
    shuffled_ids = list(id_list)
    random.shuffle(shuffled_ids)
    
    final_data = []
    successful_genomes = []
    
    # Process in batches to keep memory/queue managed
    BATCH_SIZE = n_species * 2 
    current_idx = 0
    
    pbar = tqdm(total=n_species, desc="Collecting Species")
    
    with ProcessPoolExecutor() as executor:
        while len(successful_genomes) < n_species and current_idx < len(shuffled_ids):
            
            # Prepare batch
            batch_ids = shuffled_ids[current_idx : current_idx + BATCH_SIZE]
            current_idx += len(batch_ids)
            
            futures = {}
            for i, gid in enumerate(batch_ids):
                job_seed = seed + current_idx + i 
                fut = executor.submit(process_genome, gid, directory, seqs_per_genome, seq_length, job_seed)
                futures[fut] = gid
                
            # Process batch results
            for future in as_completed(futures):
                if len(successful_genomes) >= n_species:
                    break 
                
                rows, success = future.result()
                
                if success:
                    successful_genomes.append(futures[future])
                    final_data.extend(rows)
                    pbar.update(1)
            
            futures.clear()

    pbar.close()
    
    if len(successful_genomes) < n_species:
        print(f"WARNING: Only available {len(successful_genomes)} species out of {len(id_list)} candidates fit the criteria.")
    else:
        print(f"Successfully collected {n_species} species.")

    # Create DataFrame with 2 columns
    df = pd.DataFrame(final_data, columns=["fragment_id", "seq"])
    
    return df

genome_ids = generate_genome_id_list('<GTDB dataset root>/GTDB/Bacteria/')

df_result = sample_sequences_parallel(
    directory='<GTDB dataset root>/GTDB/Bacteria/',
    id_list=genome_ids, 
    seqs_per_genome=100, 
    seq_length=50000, 
    n_species=20, 
    seed=42
)

df_result['genome_id'] = df_result['fragment_id'].apply(lambda x: x.rsplit('_', 1)[0])
df_result.to_csv('<name of output file>.csv', index=False)