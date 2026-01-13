import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import sys
import struct
from tqdm import tqdm

from utils.fasta_loader import load_fasta
from utils.protein_fvecs import save_fvecs, load_fvecs

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
        
        # Μέγιστο μήκος ακολουθίας: 1024 tokens.
        # Απαραίτητο λόγω της τετραγωνικής πολυπλοκότητας μνήμης του Attention(O(L2)).
        self.model.eval()
        self.max_length = 1022

    # Βήμα 1: Tokenization and Truncation
    # Περιορισμός μήκους ακολουθίας
    def truncate_sequence(self, sequence: str) -> str:
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
        
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator, 
                desc="Generating embeddings", 
                total=num_batches,
                unit="batch"
            )
        # Process in batches
        for i in iterator:
            batch = sequences[i:i + batch_size]
            
            batch_data = []
            batch_ids = []
            batch_lengths = []

            for protein_id, sequence in batch:
                truncated_seq = self.truncate_sequence(sequence)
                batch_data.append((protein_id, truncated_seq))
                batch_ids.append(protein_id)
                batch_lengths.append(len(truncated_seq))
            
            labels, strs, tokens = self.batch_converter(batch_data)
            tokens = tokens.to(self.device)

            # Βήμα 3:Inference
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[self.num_layers])
            
            # Βήμα 4: Mean pooling over token embeddings (excluding padding and special tokens)
            token_embeddings = results["representations"][self.num_layers]
        
            batch_embeddings = []
            for seq_idx, seq_len in enumerate(batch_lengths):
                seq_tokens = token_embeddings[seq_idx, 1:seq_len+1, :]
                seq_embedding = seq_tokens.mean(dim=0)
                batch_embeddings.append(seq_embedding)
            
            batch_embeddings = torch.stack(batch_embeddings)
            embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
            all_ids.extend(batch_ids)
        
        final_embeddings = np.vstack(all_embeddings)
        
        print(f"\nGenerated embeddings for {len(all_ids)} proteins")
        print(f"Shape: {final_embeddings.shape}")
        print(f"Dtype: {final_embeddings.dtype}")
        
        return final_embeddings, all_ids
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        ids: List[str], 
        output_path: str,
        save_fvecs_format: bool = True
    ):
        output_path = Path(output_path)
        
        # Save embeddings as .npy (Python)
        embeddings_file = output_path.with_suffix('.npy')
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to: {embeddings_file}")
        
        # Save embeddings as .fvecs (C++)
        if save_fvecs_format:
            fvecs_file = output_path.with_suffix('.fvecs')
            save_fvecs(str(fvecs_file), embeddings)
        
        # Save IDs as .txt file
        txt_file = output_path.with_suffix('.txt')
        with open(txt_file, 'w') as f:
            for protein_id in ids:
                f.write(f"{protein_id}\n")
        print(f"Saved protein IDs to: {txt_file}")
        
        # Save IDs as .ids file (backward compatibility)
        ids_file = output_path.with_suffix('.ids')
        with open(ids_file, 'w') as f:
            for protein_id in ids:
                f.write(f"{protein_id}\n")
        print(f"Saved protein IDs to: {ids_file}")
        
        # Save metadata
        metadata_file = output_path.with_suffix('.meta')
        with open(metadata_file, 'w') as f:
            f.write(f"num_proteins: {len(ids)}\n")
            f.write(f"embed_dim: {embeddings.shape[1]}\n")
            f.write(f"dtype: {embeddings.dtype}\n")
            f.write(f"model: ESM-2 ({self.num_layers} layers)\n")
        print(f"Saved metadata to: {metadata_file}")
    
    @staticmethod
    def load_embeddings(input_path: str, prefer_fvecs: bool = False) -> Tuple[np.ndarray, List[str]]:
        input_path = Path(input_path)
        
        # Try loading from .fvecs first if preferred
        fvecs_file = input_path.with_suffix('.fvecs')
        npy_file = input_path.with_suffix('.npy')
        
        if prefer_fvecs and fvecs_file.exists():
            embeddings = load_fvecs(str(fvecs_file))
            print(f"Loaded embeddings from: {fvecs_file}")
        elif npy_file.exists():
            embeddings = np.load(npy_file)
            print(f"Loaded embeddings from: {npy_file}")
        elif fvecs_file.exists():
            embeddings = load_fvecs(str(fvecs_file))
            print(f"Loaded embeddings from: {fvecs_file}")
        else:
            raise FileNotFoundError(f"No embeddings found at {input_path}")
        
        # Load IDs
        ids = None
        txt_file = input_path.with_suffix('.txt')
        ids_file = input_path.with_suffix('.ids')
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            print(f"Loaded IDs from: {txt_file}")
        elif ids_file.exists():
            with open(ids_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            print(f"Loaded IDs from: {ids_file}")
        else:
            print(f"Warning: No ID file found, using indices")
            ids = [f"protein_{i}" for i in range(len(embeddings))]
        
        print(f"Loaded {len(ids)} embeddings, shape: {embeddings.shape}")
        
        return embeddings, ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using ESM-2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input FASTA file with protein sequences')
    parser.add_argument('-o', '--output', required=True,
                        help='Output file path')
    parser.add_argument('-model', default='esm2_t6_8M_UR50D',
                        help='ESM-2 model variant (default: esm2_t6_8M_UR50D)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for embedding generation (default: 8)')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum number of sequences to process (for testing)')
    parser.add_argument('--no-fvecs', action='store_true',
                        help='Skip saving .fvecs format (C++ compatible)')
    
    args = parser.parse_args()
    
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
    print(f"C++ fvecs output: {not args.no_fvecs}")
    print("=" * 70)
    print()
    
    print("Loading FASTA sequences...")
    sequences = load_fasta(args.input)
    
    if args.max_sequences:
        print(f"Limiting to {args.max_sequences} sequences (testing mode)")
        sequences = sequences[:args.max_sequences]
    
    print(f"Loaded {len(sequences)} sequences")
    
    lengths = [len(seq) for _, seq in sequences]
    print(f"Length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    print("\n Generating embeddings...")
    embedder = ESM2Embedder(model_name=args.model, device=args.device)
    embeddings, ids = embedder.embed_sequences(
        sequences,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    print("\n Saving results...")
    embedder.save_embeddings(
        embeddings, 
        ids, 
        args.output,
        save_fvecs_format=not args.no_fvecs
    )
    
    print("\n" + "=" * 70)
    print(" Embedding generation complete")
    print("=" * 70)
    print(f"Generated {len(ids)} embeddings of dimension {embeddings.shape[1]}")
    print(f"\nFiles created:")
    print(f"  - {args.output}.npy   (embeddings - Python)")
    if not args.no_fvecs:
        print(f"  - {args.output}.fvecs (embeddings - C++)")
    print(f"  - {args.output}.txt   (protein IDs)")
    print(f"  - {args.output}.ids   (protein IDs)")
    print(f"  - {args.output}.meta  (metadata)")
    print()


if __name__ == '__main__':
    main()