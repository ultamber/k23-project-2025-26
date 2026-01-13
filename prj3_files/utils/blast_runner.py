import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pickle


class BLASTRunner:

    def __init__(
        self,
        db_fasta: Optional[str] = None,
        makeblastdb_path: str = 'makeblastdb',
        blastp_path: str = 'blastp',
        evalue_threshold: float = 0.01
    ):

        self.db_fasta = db_fasta
        self.makeblastdb_path = makeblastdb_path
        self.blastp_path = blastp_path
        self.evalue_threshold = evalue_threshold
        
        self.db_path = None
        self.temp_dir = None
    
    def build_database(self, fasta_file: str, output_db: Optional[str] = None):

        if output_db is None:
            # Use temp directory with proper cleanup
            self.temp_dir = tempfile.mkdtemp(prefix='blast_db_')
            output_db = str(Path(self.temp_dir) / 'blast_db')
        
        self.db_path = output_db
        
        print(f"Building BLAST database...")
        print(f"Input: {fasta_file}")
        print(f"Output: {output_db}")
        
        cmd = [
            self.makeblastdb_path,
            '-in', fasta_file,
            '-dbtype', 'prot',
            '-out', output_db
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"BLAST database created")
            
        except subprocess.CalledProcessError as e:
            print(f"makeblastdb failed:")
            print(e.stderr)
            raise
        except FileNotFoundError:
            raise RuntimeError(
                f"makeblastdb not found at: {self.makeblastdb_path}\n"
                f"Install BLAST+ tools or provide correct path"
            )
    
    def search_fasta(
        self,
        query_fasta: str,
        N: int = 50,
        num_threads: int = 8,
        max_target_seqs: int = 100
    ) -> Dict[str, List[Tuple[str, float, float]]]:

        if self.db_path is None:
            if self.db_fasta is None:
                raise RuntimeError("No database specified")
            self.build_database(self.db_fasta)
        
        print(f"\nRunning BLAST search...")
        print(f"Query: {query_fasta}")
        print(f"Database: {self.db_path}")
        print(f"N: {N}")
        print(f"Threads: {num_threads}")
        
        # Create temp output file
        output_fd, output_file = tempfile.mkstemp(prefix='blast_results_', suffix='.tsv')
        
        # BLAST command with tabular output
        # Format 6 fields: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
        cmd = [
            self.blastp_path,
            '-db', self.db_path,
            '-query', query_fasta,
            '-out', output_file,
            '-outfmt', '6 qseqid sseqid pident length evalue bitscore',
            '-evalue', str(self.evalue_threshold),
            '-num_threads', str(num_threads),
            '-max_target_seqs', str(max_target_seqs)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            print(f"BLAST search completed")
            
            # Parse results
            results = self._parse_blast_output(output_file, N)
            
            # Cleanup
            Path(output_file).unlink(missing_ok=True)
            
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"BLAST search failed:")
            print(e.stderr)
            raise
        except subprocess.TimeoutExpired:
            print(f"BLAST search timed out (>1 hour)")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                f"blastp not found at: {self.blastp_path}\n"
                f"Install BLAST+ tools or provide correct path"
            )
    
    def _parse_blast_output(
        self,
        output_file: str,
        N: int
        ) -> Dict[str, List[Tuple[str, float, float, float]]]:

        results = {}
        
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 6:
                    continue
                
                query_id = parts[0]
                subject_id = parts[1]
                pident = float(parts[2])
                length = int(parts[3])
                evalue = float(parts[4])
                bitscore = float(parts[5])
                
                # Skip self-hits
                if query_id == subject_id:
                    continue
                
                if evalue > self.evalue_threshold:
                    continue
                
                if query_id not in results:
                    results[query_id] = []
                
                results[query_id].append((subject_id, bitscore, evalue, pident))
        
        for query_id in results:
            results[query_id].sort(key=lambda x: x[1], reverse=True)
            results[query_id] = results[query_id][:N]
        
        print(f"Parsed results for {len(results)} queries (with identity %)")
        
        return results
    
    def results_to_id_lists(
        self,
        results: Dict[str, List[Tuple[str, float, float]]],
        id_to_index: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[int]]:

        if id_to_index is None:
            # Assume IDs are already indices
            return {
                qid: [int(hit[0]) for hit in hits]
                for qid, hits in results.items()
            }
        
        # Map IDs to indices
        id_lists = {}
        for query_id, hits in results.items():
            indices = []
            for hit_id, _, _ in hits:
                if hit_id in id_to_index:
                    indices.append(id_to_index[hit_id])
            id_lists[query_id] = indices
        
        return id_lists
    
    def save_results(self, results: Dict, output_file: str):

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"BLAST results saved to: {output_file}")
    
    @staticmethod
    def load_results(input_file: str) -> Dict:

        with open(input_file, 'rb') as f:
            results = pickle.load(f)
        
        print(f"BLAST results loaded from: {input_file}")
        return results
    
    def cleanup(self):
        if self.db_path:
            # Remove database files
            for ext in ['.phr', '.pin', '.psq', '.pdb', '.pot', '.ptf', '.pto']:
                db_file = Path(f"{self.db_path}{ext}")
                db_file.unlink(missing_ok=True)
            
            # Remove temp directory if created
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir)
            
            print(f"Cleaned up BLAST database")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run BLAST and save results for ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-d', '--database',
        required=True,
        help='Database FASTA file'
    )
    parser.add_argument(
        '-q', '--queries',
        required=True,
        help='Query FASTA file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output pickle file'
    )
    
    # Optional ID mapping (for embeddings)
    parser.add_argument(
        '--database-ids',
        help='Database protein IDs file (.ids from protein_embed.py)'
    )
    parser.add_argument(
        '--query-ids',
        help='Query protein IDs file (.ids from protein_embed.py)'
    )
    
    # BLAST parameters
    parser.add_argument(
        '--N',
        type=int,
        default=50,
        help='Number of top hits to keep per query (default: 50)'
    )
    parser.add_argument(
        '--evalue',
        type=float,
        default=0.01,
        help='E-value threshold (default: 0.01)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of CPU threads (default: 8)'
    )
    parser.add_argument(
        '--max-target-seqs',
        type=int,
        default=100,
        help='Maximum target sequences (default: 100)'
    )
    
    # BLAST binary paths
    parser.add_argument(
        '--makeblastdb',
        default='makeblastdb',
        help='Path to makeblastdb binary'
    )
    parser.add_argument(
        '--blastp',
        default='blastp',
        help='Path to blastp binary'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Blast runner")
    print("="*70)

    db_id_to_index = None
    query_id_to_index = None
    
    if args.database_ids or args.query_ids:
        print(f"\nLoading ID mappings...")
        
        if args.database_ids:
            with open(args.database_ids, 'r') as f:
                db_ids = [line.strip() for line in f]
            db_id_to_index = {pid: i for i, pid in enumerate(db_ids)}
            print(f"Database IDs: {len(db_ids)} proteins")
        
        if args.query_ids:
            with open(args.query_ids, 'r') as f:
                query_ids = [line.strip() for line in f]
            query_id_to_index = {pid: i for i, pid in enumerate(query_ids)}
            print(f"Query IDs: {len(query_ids)} proteins")

    print(f"\nInitializing BLAST...")
    
    try:
        blast = BLASTRunner(
            db_fasta=args.database,
            makeblastdb_path=args.makeblastdb,
            blastp_path=args.blastp,
            evalue_threshold=args.evalue
        )
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure BLAST+ is installed:")
        print("  Ubuntu/Debian: sudo apt-get install ncbi-blast+")
        print("  macOS: brew install blast")
        print("  Or download from: https://blast.ncbi.nlm.nih.gov/")
        exit(1)

    print(f"\nBuilding BLAST database...")
    blast.build_database(args.database)
    
    print(f"\nRunning BLAST search...")
    print(f"Query file: {args.queries}")
    print(f"Top N: {args.N}")
    print(f"E-value threshold: {args.evalue}")
    print(f"Threads: {args.threads}")
    print(f"Max target seqs: {args.max_target_seqs}")
    
    try:
        results = blast.search_fasta(
            args.queries,
            N=args.N,
            num_threads=args.threads,
            max_target_seqs=args.max_target_seqs
        )
    except Exception as e:
        print(f"\nBLAST search failed: {e}")
        blast.cleanup()
        exit(1)

    print(f"\nProcessing results...")
    
    # Prepare output data
    output_data = {
        'blast_results_ids': results,
        'params': {
            'N': args.N,
            'evalue': args.evalue,
            'database': str(args.database),
            'queries': str(args.queries),
            'threads': args.threads,
        }
    }
    
    # Convert to indices if mappings available
    if db_id_to_index and query_id_to_index:
        print("  Converting to index-based format...")
        
        # Import get_accession if available
        try:
            from utils.fasta_loader import get_accession
        except:
            # Fallback: use ID as-is
            get_accession = lambda x: x.split('|')[1] if '|' in x else x
        
        blast_results_indices = {}
        conversion_stats = {
            'total_queries': 0,
            'converted_queries': 0,
            'missing_query_ids': [],
            'missing_hit_ids': set()
        }
        
        for query_id, hits in results.items():
            conversion_stats['total_queries'] += 1
            
            # Get query index
            query_acc = get_accession(query_id)
            
            if query_acc not in query_id_to_index:
                conversion_stats['missing_query_ids'].append(query_acc)
                continue
            
            query_idx = query_id_to_index[query_acc]
            
            # Convert hit IDs to indices
            hit_indices = []
            for hit in hits:
                if len(hit) >= 4:
                    hit_id, score, evalue, pident = hit[0], hit[1], hit[2], hit[3]
                else:
                    hit_id, score, evalue = hit[0], hit[1], hit[2]
                    pident = None
                hit_acc = get_accession(hit_id)
                
                if hit_acc in db_id_to_index:
                    hit_idx = db_id_to_index[hit_acc]
                    hit_indices.append((hit_idx, score, evalue))
                else:
                    conversion_stats['missing_hit_ids'].add(hit_acc)
            
            if hit_indices:  # Only add if we found some hits
                blast_results_indices[query_idx] = hit_indices
                conversion_stats['converted_queries'] += 1
        
        output_data['blast_results_indices'] = blast_results_indices
        
        # Print conversion stats
        print(f"Converted {conversion_stats['converted_queries']}/{conversion_stats['total_queries']} queries")
        
        if conversion_stats['missing_query_ids']:
            print(f"{len(conversion_stats['missing_query_ids'])} query IDs not found in mapping")
            if len(conversion_stats['missing_query_ids']) <= 5:
                for qid in conversion_stats['missing_query_ids']:
                    print(f"- {qid}")
        
        if conversion_stats['missing_hit_ids']:
            print(f"{len(conversion_stats['missing_hit_ids'])} hit IDs not found in mapping")
            if len(conversion_stats['missing_hit_ids']) <= 5:
                for hid in list(conversion_stats['missing_hit_ids'])[:5]:
                    print(f"- {hid}")
    
    print(f"\nSaving results...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Results saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Queries processed: {len(results)}")
    print(f"Top-N per query: {args.N}")
    print(f"E-value threshold: {args.evalue}")
    
    # Print sample results
    if results:
        sample_query = list(results.keys())[0]
        sample_hits = results[sample_query][:5]
        
        print(f"\nSample results (query: {sample_query}):")
        for hit in sample_hits:
            hit_id = hit[0]
            score = hit[1]
            evalue = hit[2]
            pident = hit[3] if len(hit) > 3 else None
            print(f"{hit_id}: score={score:.1f}, E={evalue:.2e}")
    
    # Data format info
    print(f"\nOutput format:")
    print(f"- blast_results_ids: ID-based results")
    if 'blast_results_indices' in output_data:
        print(f"- blast_results_indices: Index-based results")
    print(f"- params: Search parameters")
    
    print(f"\nUsage in protein_search.py:")
    print(f"python protein_search.py \\")
    print(f"--ground-truth {args.output} \\")
    print(f"--embeddings <embeddings.npy> \\")
    print(f"--queries <queries.npy>")
    
    # Cleanup
    blast.cleanup()
    
    print(f"\n{'='*70}")
    print("BLAST SEARCH COMPLETED")
    print(f"{'='*70}\n")