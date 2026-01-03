import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import sys
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.fasta_loader import load_fasta, get_accession


class ESM2Embedder:
    
    def __init__(self, model_name: str = "esm2_t6_8M_UR50D", device: str = "auto"):
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
                
        try:
            import esm
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = model.to(self.device)
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
            
            self.num_layers = model.num_layers
            self.embed_dim = model.embed_dim
            
            print(f"Model loaded: {self.num_layers} layers, {self.embed_dim} dimensions")
            
        except ImportError:
            print("\n ERROR: ESM library not found.")
            print("Please install: pip install fair-esm")
            sys.exit(1)
        
        self.model.eval()
        
        # Μέγιστο μήκος ακολουθίας: 1024 tokens.
        # Απαραίτητο λόγω της τετραγωνικής πολυπλοκότητας μνήμης του Attention(O(L2)).
        self.max_length = 1022
    
    # Βήμα 1: Tokenization and Truncation
    def truncate_sequence(self, sequence: str) -> str:

        # Περιορισμός μήκους ακολουθίας
        if len(sequence) > self.max_length:
            print(f"Warning: Truncating sequence of length {len(sequence)} to {self.max_length}")
            return sequence[:self.max_length]
        return sequence
    
    # Βήμα 2: Batching 
    def embed_sequences(
        self, 
        sequences: List[Tuple[str, str]], 
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:

        all_embeddings = []
        all_ids = []
        
        # Process in batches
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator, 
                desc="Generating embeddings", 
                total=num_batches,
                unit="batch"
            )
        
        for i in iterator:
            batch = sequences[i:i + batch_size]
            
            # Prepare batch data
            batch_data = []
            batch_ids = []
            
            for protein_id, sequence in batch:
                # Truncate if necessary
                truncated_seq = self.truncate_sequence(sequence)
                batch_data.append((protein_id, truncated_seq))
                batch_ids.append(protein_id)
            
            # Convert batch
            labels, strs, tokens = self.batch_converter(batch_data)
            tokens = tokens.to(self.device)
            
            # Βήμα 3:Inference
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[self.num_layers])
            
            # Extract last layer's representations (το οποίο περιέχει την πιο αφηρημένη/σημασιολογική πληροφορία της ακολουθίας)
            # Shape: (batch_size, seq_len, embed_dim)
            token_embeddings = results["representations"][self.num_layers]
            
            # Βήμα 4: Mean pooling
            # TODO fix mean pooling over valid tokens ??????
            embedding = token_embeddings.mean(dim=1)
            # Convert to numpy and store
            embeddings_np = embedding.cpu().numpy()
            all_embeddings.append(embeddings_np)
            all_ids.extend(batch_ids)
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings)
        
        print(f"\nGenerated embeddings for {len(all_ids)} proteins")
        print(f"Shape: {final_embeddings.shape}")
        print(f"Dtype: {final_embeddings.dtype}")
        
        return final_embeddings, all_ids
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        ids: List[str], 
        output_path: str
    ):
        
        output_path = Path(output_path)
        
        # Save embeddings as .npy
        # Numpy binary : vector array of dimensions N x 320 (where N is number of proteins)
        embeddings_file = output_path.with_suffix('.npy')
        np.save(embeddings_file, embeddings)
        print(f" Saved embeddings to: {embeddings_file}")
        
        # Save IDs as txt file
        # Index mapping : text file that matches each row in the vector array to the corresponding protein ID
        ids_file = output_path.with_suffix('.ids')
        with open(ids_file, 'w') as f:
            for protein_id in ids:
                f.write(f"{protein_id}\n")
        print(f" Saved protein IDs to: {ids_file}")
        
        # Save metadata
        metadata_file = output_path.with_suffix('.meta')
        with open(metadata_file, 'w') as f:
            f.write(f"num_proteins: {len(ids)}\n")
            f.write(f"embed_dim: {embeddings.shape[1]}\n")
            f.write(f"dtype: {embeddings.dtype}\n")
            f.write(f"model: ESM-2 ({self.num_layers} layers)\n")
        print(f" Saved metadata to: {metadata_file}")
    
    @staticmethod
    def load_embeddings(input_path: str) -> Tuple[np.ndarray, List[str]]:

        input_path = Path(input_path)
        
        # Load embeddings
        embeddings_file = input_path.with_suffix('.npy')
        embeddings = np.load(embeddings_file)
        
        # Load IDs
        ids_file = input_path.with_suffix('.ids')
        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f]
        
        print(f" Loaded {len(ids)} embeddings from {input_path}")
        print(f"Shape: {embeddings.shape}")
        
        return embeddings, ids


def main():

    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using ESM-2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input FASTA file with protein sequences'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file path'
    )
    
    # Model options
    parser.add_argument(
        '-model',
        default='esm2_t6_8M_UR50D',
        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D'],
        help='ESM-2 model variant (default: esm2_t6_8M_UR50D)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for embedding generation (default: 8)'
    )
    
    # Processing options
    parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum number of sequences to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f" Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("=" * 70)
    print("Protein Embedding Generation using ESM-2")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch:  {args.batch_size}")
    print("=" * 70)
    print()
    
    # Step 1: Load sequences
    print("Loading FASTA sequences...")
    sequences = load_fasta(args.input)
    
    if args.max_sequences:
        print(f"Limiting to {args.max_sequences} sequences (testing mode)")
        sequences = sequences[:args.max_sequences]
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Show sequence length statistics
    lengths = [len(seq) for _, seq in sequences]
    print(f"Length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # Step 2: Generate embeddings
    print("\n Generating embeddings...")
    embedder = ESM2Embedder(model_name=args.model, device=args.device)
    embeddings, ids = embedder.embed_sequences(
        sequences,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    # Step 3: Save results
    print("\n Saving results...")
    embedder.save_embeddings(embeddings, ids, args.output)
    
    print("\n" + "=" * 70)
    print(" Embedding generation complete")
    print("=" * 70)
    print(f"Generated {len(ids)} embeddings of dimension {embeddings.shape[1]}")
    print(f"\nFiles created:")
    print(f"- {args.output}.npy  (embeddings)")
    print(f"- {args.output}.ids  (protein IDs)")
    print(f"- {args.output}.meta (metadata)")
    print()


if __name__ == '__main__':
    main()