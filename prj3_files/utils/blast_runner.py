import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle


def get_accession(protein_id: str) -> str:
    """
    Extract accession from various protein ID formats.

    Handles:
    - sp|Q5F928|UVRC_NEIG1 -> Q5F928
    - tr|A0A009I3Y5|A0A009I3Y5_ACIBA -> A0A009I3Y5
    - Q5F928 -> Q5F928
    - A0A009I3Y5 ACIBA -> A0A009I3Y5
    """
    protein_id = protein_id.strip()

    # UniProt format: sp|ACCESSION|ENTRY_NAME or tr|ACCESSION|ENTRY_NAME
    if '|' in protein_id:
        parts = protein_id.split('|')
        if len(parts) >= 2:
            return parts[1]

    # Space-separated: ACCESSION DESCRIPTION
    if ' ' in protein_id:
        return protein_id.split()[0]

    return protein_id


def build_id_mapping(ids: List[str]) -> Dict[str, int]:
    id_to_index = {}

    for i, pid in enumerate(ids):
        pid = pid.strip()

        # Add full ID
        id_to_index[pid] = i
        id_to_index[pid.split()[0]] = i  # Add accession if space-separated

        if '|' in pid:
            parts = pid.split('|')

            # Add accession (e.g., Q5F928)
            if len(parts) >= 2:
                id_to_index[parts[1]] = i

            # Add entry name (e.g., UVRC_NEIG1)
            if len(parts) >= 3:
                entry_name = parts[2].split()[0]  # Remove description if present
                id_to_index[entry_name] = i
        else:
            # Simple format: just accession or accession + description
            accession = pid.split()[0]
            id_to_index[accession] = i

    return id_to_index


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
            self.temp_dir = tempfile.mkdtemp(prefix='blast_db_')
            output_db = str(Path(self.temp_dir) / 'blast_db')

        self.db_path = output_db

        print(f"Building BLAST database...")
        print(f"  Input: {fasta_file}")
        print(f"  Output: {output_db}")

        cmd = [
            self.makeblastdb_path,
            '-in', fasta_file,
            '-dbtype', 'prot',
            '-out', output_db
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  BLAST database created")
        except subprocess.CalledProcessError as e:
            print(f"  makeblastdb failed: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(f"makeblastdb not found at: {self.makeblastdb_path}")

    def search_fasta(
        self,
        query_fasta: str,
        N: int = 50,
        num_threads: int = 8,
        max_target_seqs: int = 100
    ) -> Dict[str, List[Tuple[str, float, float, float]]]:

        if self.db_path is None:
            if self.db_fasta is None:
                raise RuntimeError("No database specified")
            self.build_database(self.db_fasta)

        print(f"\nRunning BLAST search...")
        print(f"  Query: {query_fasta}")
        print(f"  Database: {self.db_path}")
        print(f"  N: {N}, Threads: {num_threads}")

        output_fd, output_file = tempfile.mkstemp(prefix='blast_results_', suffix='.tsv')

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
        os.close(output_fd)
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
            print(f"  BLAST search completed")
            results = self._parse_blast_output(output_file, N)
            Path(output_file).unlink(missing_ok=True)
            return results

        except subprocess.CalledProcessError as e:
            print(f"  BLAST search failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            print(f"  BLAST search timed out")
            raise


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
                if get_accession(query_id) == get_accession(subject_id):
                    continue

                if evalue > self.evalue_threshold:
                    continue

                if query_id not in results:
                    results[query_id] = []

                # Store all 4 values including pident
                results[query_id].append((subject_id, bitscore, evalue, pident))

        # Sort by bitscore and keep top N
        for query_id in results:
            results[query_id].sort(key=lambda x: x[1], reverse=True)
            results[query_id] = results[query_id][:N]

        print(f"  Parsed results for {len(results)} queries")
        return results

    def convert_to_indices(
        self,
        results: Dict[str, List[Tuple]],
        db_id_to_index: Dict[str, int],
        query_id_to_index: Dict[str, int],
        verbose: bool = True
    ) -> Dict[int, List[Tuple]]:

        blast_indices = {}

        stats = {
            'total_queries': 0,
            'converted_queries': 0,
            'total_hits': 0,
            'converted_hits': 0,
            'missing_queries': [],
            'missing_hits': set()
        }

        for query_id, hits in results.items():
            stats['total_queries'] += 1

            # Try multiple forms of query ID
            query_idx = None
            query_acc = get_accession(query_id)

            for qid_form in [query_id, query_acc]:
                if qid_form in query_id_to_index:
                    query_idx = query_id_to_index[qid_form]
                    break

            if query_idx is None:
                stats['missing_queries'].append(query_id)
                continue

            # Convert hits
            hit_indices = []
            for hit in hits:
                stats['total_hits'] += 1

                hit_id = hit[0]
                bitscore = hit[1]
                evalue = hit[2]
                pident = hit[3] if len(hit) > 3 else None

                # Try multiple forms of hit ID
                hit_idx = None
                hit_acc = get_accession(hit_id)

                for hid_form in [hit_id, hit_acc]:
                    if hid_form in db_id_to_index:
                        hit_idx = db_id_to_index[hid_form]
                        break

                if hit_idx is not None:
                    stats['converted_hits'] += 1
                    # Include pident in the tuple!
                    if pident is not None:
                        hit_indices.append((hit_idx, bitscore, evalue, pident))
                    else:
                        hit_indices.append((hit_idx, bitscore, evalue))
                else:
                    stats['missing_hits'].add(hit_id)

            if hit_indices:
                blast_indices[query_idx] = hit_indices
                stats['converted_queries'] += 1

        if verbose:
            print(f"\n  Conversion stats:")
            print(f"    Queries: {stats['converted_queries']}/{stats['total_queries']}")
            print(f"    Hits: {stats['converted_hits']}/{stats['total_hits']}")

            if stats['missing_queries']:
                print(f"    Missing queries: {len(stats['missing_queries'])}")
                if len(stats['missing_queries']) <= 3:
                    for qid in stats['missing_queries']:
                        print(f"      - {qid}")

            if stats['missing_hits']:
                print(f"    Missing hits: {len(stats['missing_hits'])}")
                if len(stats['missing_hits']) <= 3:
                    for hid in list(stats['missing_hits'])[:3]:
                        print(f"      - {hid}")

        return blast_indices

    def cleanup(self):
        if self.db_path:
            for ext in ['.phr', '.pin', '.psq', '.pdb', '.pot', '.ptf', '.pto']:
                db_file = Path(f"{self.db_path}{ext}")
                db_file.unlink(missing_ok=True)

            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir)


def run_blast_and_convert(
    database_fasta: str,
    query_fasta: str,
    database_ids: List[str],
    query_ids: List[str],
    N: int = 50,
    evalue: float = 0.01,
    threads: int = 8,
    verbose: bool = True
) -> Dict:
    """
    Complete BLAST workflow: search and convert to indices.

    This is the function to call from protein_search.py.

    Returns:
        Dict with:
        - 'blast_results_ids': Original ID-based results
        - 'blast_results_indices': Index-based results for recall calculation
        - 'params': Search parameters
    """

    # Build comprehensive ID mappings
    if verbose:
        print(f"\nBuilding ID mappings...")
        print(f"  Database: {len(database_ids)} proteins")
        print(f"  Queries: {len(query_ids)} proteins")

    db_id_to_index = build_id_mapping(database_ids)
    query_id_to_index = build_id_mapping(query_ids)

    if verbose:
        print(f"  Database mapping entries: {len(db_id_to_index)}")
        print(f"  Query mapping entries: {len(query_id_to_index)}")

        # Debug: show sample mappings
        sample_db_ids = list(database_ids)[:2]
        for sid in sample_db_ids:
            acc = get_accession(sid)
            print(f"    DB sample: '{sid}' -> acc='{acc}'")

    # Run BLAST
    blast = BLASTRunner(evalue_threshold=evalue)
    blast.build_database(database_fasta)

    results_ids = blast.search_fasta(
        query_fasta=query_fasta,
        N=N,
        num_threads=threads
    )

    # Debug: show sample BLAST results
    if verbose and results_ids:
        sample_qid = list(results_ids.keys())[0]
        sample_hits = results_ids[sample_qid][:2]
        print(f"\n  Sample BLAST results:")
        print(f"    Query: '{sample_qid}' -> acc='{get_accession(sample_qid)}'")
        for hit in sample_hits:
            hit_id = hit[0]
            hit_acc = get_accession(hit_id)
            in_db = hit_acc in db_id_to_index
            print(f"    Hit: '{hit_id}' -> acc='{hit_acc}' in_db={in_db}")

    # Convert to indices
    results_indices = blast.convert_to_indices(
        results=results_ids,
        db_id_to_index=db_id_to_index,
        query_id_to_index=query_id_to_index,
        verbose=verbose
    )

    blast.cleanup()

    return {
        'blast_results_ids': results_ids,
        'blast_results_indices': results_indices,
        'params': {
            'N': N,
            'evalue': evalue,
            'database_fasta': database_fasta,
            'query_fasta': query_fasta
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run BLAST and save results")
    parser.add_argument('-d', '--database', required=True, help='Database FASTA')
    parser.add_argument('-q', '--queries', required=True, help='Query FASTA')
    parser.add_argument('-o', '--output', required=True, help='Output pickle file')
    parser.add_argument('--database-ids', help='Database protein IDs file')
    parser.add_argument('--query-ids', help='Query protein IDs file')
    parser.add_argument('--N', type=int, default=50, help='Top N hits')
    parser.add_argument('--evalue', type=float, default=0.01, help='E-value threshold')
    parser.add_argument('--threads', type=int, default=8, help='CPU threads')

    args = parser.parse_args()

    print("="*70)
    print("BLAST Runner (Fixed)")
    print("="*70)

    # Load ID mappings
    database_ids = None
    query_ids = None

    if args.database_ids:
        with open(args.database_ids, 'r') as f:
            database_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(database_ids)} database IDs")

    if args.query_ids:
        with open(args.query_ids, 'r') as f:
            query_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(query_ids)} query IDs")

    # Run BLAST
    if database_ids and query_ids:
        overlap = set(map(get_accession, database_ids)) & set(map(get_accession, query_ids))
        if overlap:
            print(f"WARNING: {len(overlap)} query accessions also appear in database (possible leakage)")
        results = run_blast_and_convert(
            database_fasta=args.database,
            query_fasta=args.queries,
            database_ids=database_ids,
            query_ids=query_ids,
            N=args.N,
            evalue=args.evalue,
            threads=args.threads
        )
    else:
        # Run without ID conversion
        blast = BLASTRunner(evalue_threshold=args.evalue)
        blast.build_database(args.database)
        results_ids = blast.search_fasta(args.queries, N=args.N, num_threads=args.threads)
        blast.cleanup()

        results = {
            'blast_results_ids': results_ids,
            'params': {'N': args.N, 'evalue': args.evalue}
        }

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {args.output}")

    # Summary
    print(f"\nSummary:")
    print(f"  Queries with results: {len(results.get('blast_results_ids', {}))}")
    if 'blast_results_indices' in results:
        print(f"  Queries converted to indices: {len(results['blast_results_indices'])}")

    print("\nDone!")
