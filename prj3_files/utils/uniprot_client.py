import requests
import time
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# https://www.uniprot.org/api-documentation/uniprotkb
# Uniprot REST API client documentation
class UniProtClient:

    BASE_URL = "https://rest.uniprot.org/uniprotkb"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RemoteHomologs/1.0'
        })

    def get_annotation(self, protein_id: str, use_cache: bool = True) -> Optional[Dict]:

        # Check cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{protein_id}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # Fetch from API
        url = f"{self.BASE_URL}/{protein_id}.json"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Cache response
                if self.cache_dir:
                    cache_file = self.cache_dir / f"{protein_id}.json"
                    with open(cache_file, 'w') as f:
                        json.dump(data, f, indent=2)

                return data

            elif response.status_code == 404:
                return None

            else:
                print(f"Warning: API returned {response.status_code} for {protein_id}")
                return None

        except Exception as e:
            print(f"Error fetching {protein_id}: {e}")
            return None

    def extract_go_terms(self, annotation: Dict) -> List[Tuple[str, str, str]]:
        go_terms = []

        if not annotation:
            return go_terms
        refs = annotation.get('dbReferences') or annotation.get('uniProtKBCrossReferences') or []

        # UniProt JSON format: uniProtKBCrossReferences with database="GO"
        for ref in refs:
            if ref.get('database') == 'GO':
                go_id = ref.get('id', '')

                # Get properties
                properties = {prop['key']: prop['value'] 
                             for prop in ref.get('properties', [])}

                term = properties.get('GoTerm', '')

                # Determine aspect from term prefix
                if term.startswith('P:'):
                    aspect = 'P'
                    term = term[2:]
                elif term.startswith('F:'):
                    aspect = 'F'
                    term = term[2:]
                elif term.startswith('C:'):
                    aspect = 'C'
                    term = term[2:]
                else:
                    aspect = 'Unknown'

                go_terms.append((go_id, aspect, term))

        return go_terms

    def extract_ec_numbers(self, annotation: Dict) -> List[str]:

        ec_numbers = []

        if not annotation:
            return ec_numbers

        # From protein description
        protein = annotation.get('proteinDescription', {})
        rec_name = protein.get('recommendedName', {})

        for ec in rec_name.get('ecNumbers', []):
            ec_value = ec.get('value', '')
            # Get ECO code from evidences
            evidences = ec.get('evidences', [])
            eco_code = ''
            if evidences:
                eco_code = evidences[0].get('evidenceCode', '')
            if eco_code:
                ec_numbers.append(f"{ec_value} ({eco_code})")
            else:
                ec_numbers.append(ec_value)

        # From alternative names
        for alt_name in protein.get('alternativeNames', []):
            for ec in alt_name.get('ecNumbers', []):
                ec_value = ec.get('value', '')
                evidences = ec.get('evidences', [])
                eco_code = ''
                if evidences:
                    eco_code = evidences[0].get('evidenceCode', '')
                if eco_code:
                    ec_numbers.append(f"{ec_value} ({eco_code})")
                else:
                    ec_numbers.append(ec_value)

        return list(set(ec_numbers))  # Remove duplicates

    def extract_pfam_domains(self, annotation: Dict) -> List[Tuple[str, str]]:

        pfam_domains = []

        if not annotation:
            return pfam_domains

        # UniProt JSON format: uniProtKBCrossReferences with database="Pfam"
        for ref in annotation.get('uniProtKBCrossReferences', []):
            if ref.get('database') == 'Pfam':
                pfam_id = ref.get('id', '')

                # Get properties
                properties = {prop['key']: prop['value'] 
                             for prop in ref.get('properties', [])}

                entry_name = properties.get('EntryName', '')

                pfam_domains.append((pfam_id, entry_name))

        return pfam_domains

    def extract_function(self, annotation: Dict) -> Optional[str]:
        if not annotation:
            return None

        comments = annotation.get('comments', [])

        for comment in comments:
            if comment.get('type') == 'FUNCTION':
                texts = comment.get('texts', [])
                if texts:
                    return texts[0].get('value', '')

        return None

    def extract_protein_name(self, annotation: Dict) -> Optional[str]:
        if not annotation:
            return None

        protein = annotation.get('protein', {})
        rec_name = protein.get('recommendedName', {})
        full_name = rec_name.get('fullName', {})

        return full_name.get('value')

    def extract_organism(self, annotation: Dict) -> Optional[str]:
        if not annotation:
            return None

        organism = annotation.get('organism', {})
        scientific_name = organism.get('scientificName', '')
        evidences = organism.get('evidences', [])
        eco_code = ''
        if evidences:
            eco_code = evidences[0].get('evidenceCode', '')
        
        if eco_code:
            return f"{scientific_name} ({eco_code})"
        else:
            return scientific_name

    def get_protein_summary(self, protein_id: str) -> Dict:

        annotation = self.get_annotation(protein_id)

        if not annotation:
            return {
                'id': protein_id,
                'found': False
            }

        return {
            'id': protein_id,
            'found': True,
            'name': self.extract_protein_name(annotation),
            'organism': self.extract_organism(annotation),
            'function': self.extract_function(annotation),
            'go_terms': self.extract_go_terms(annotation),
            'ec_numbers': self.extract_ec_numbers(annotation),
            'pfam_domains': self.extract_pfam_domains(annotation)
        }


    def compare_annotations(self, id1: str, id2: str) -> Dict:
        summary1 = self.get_protein_summary(id1)
        summary2 = self.get_protein_summary(id2)

        if not summary1['found'] or not summary2['found']:
            return {
                'comparable': False,
                'reason': 'One or both proteins not found'
            }

        # Compare GO terms
        go1 = set([go_id for go_id, _, _ in summary1['go_terms']])
        go2 = set([go_id for go_id, _, _ in summary2['go_terms']])
        go_common = go1 & go2
        go_only1 = go1 - go2
        go_only2 = go2 - go1

        # Compare EC numbers
        ec1 = set(summary1['ec_numbers'])
        ec2 = set(summary2['ec_numbers'])
        ec_common = ec1 & ec2

        # Compare Pfam domains
        pfam1 = set([pfam_id for pfam_id, _ in summary1['pfam_domains']])
        pfam2 = set([pfam_id for pfam_id, _ in summary2['pfam_domains']])
        pfam_common = pfam1 & pfam2

        # Similarity scores
        go_similarity = len(go_common) / max(len(go1 | go2), 1)
        pfam_similarity = len(pfam_common) / max(len(pfam1 | pfam2), 1)
        ec_match = len(ec_common) > 0

        return {
            'comparable': True,
            'protein1': summary1,
            'protein2': summary2,
            'go_common': list(go_common),
            'go_only1': list(go_only1),
            'go_only2': list(go_only2),
            'go_similarity': go_similarity,
            'ec_common': list(ec_common),
            'ec_match': ec_match,
            'pfam_common': list(pfam_common),
            'pfam_similarity': pfam_similarity,
            'has_common_function': pfam_similarity > 0.3 or go_similarity > 0.3 or ec_match
        }

def _extract_uniprot_acc(pid: str) -> str:
    token = pid.split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 3:
            return parts[1]
    return token

def _format_go_terms(go_terms, max_terms: int = 3) -> str:
    if not go_terms:
        return "GO: N/A"
    # Prioritize Function, then Process, then Component
    order = {"F": 0, "P": 1, "C": 2, "Unknown": 3}
    go_terms = sorted(go_terms, key=lambda x: order.get(x[1], 9))
    short = []
    for go_id, aspect, term in go_terms[:max_terms]:
        prefix = aspect if aspect in ("F", "P", "C") else "?"
        short.append(f"{prefix}:{term}")
    return "GO: " + "; ".join(short)

def get_uniprot_summary_cached(
    neighbour_acc: str,
    query_acc: str,
    uniprot_client,  # type: UniProtClient
    cache: Dict[str, Dict],
    state: Dict[str, float],
    delay: float = 0.2,
) -> Optional[Dict]:
    """
    state should contain: {"last_call": 0.0}
    """
    if not uniprot_client or not neighbour_acc or not query_acc:
        return None

    if neighbour_acc in cache:
        return cache[neighbour_acc]

    # rate limit
    last_call = state.get("last_call", 0.0)
    if delay and delay > 0:
        now = time.time()
        sleep_for = (last_call + delay) - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        state["last_call"] = time.time()

    summary = uniprot_client.get_protein_summary(neighbour_acc)
    cache[neighbour_acc] = summary
    return summary

def batch_fetch_annotations(
    protein_ids: List[str],
    cache_dir: Optional[str] = None,
    delay: float = 0.5,
    verbose: bool = False
) -> Dict[str, Dict]:

    client = UniProtClient(cache_dir=cache_dir)
    results = {}

    for i, protein_id in enumerate(protein_ids, 1):
        if verbose:
            print(f"[{i}/{len(protein_ids)}] Fetching {protein_id}...")

        results[protein_id] = client.get_protein_summary(protein_id)

        # Rate limiting
        if i < len(protein_ids):
            time.sleep(delay)

    return results


if __name__ == '__main__':
    """Test UniProt client."""

    # Test with hemoglobin proteins
    client = UniProtClient(cache_dir='uniprot_cache')

    print("Testing UniProt API client...\n")

    # Test single protein
    protein_id = "A0A009HN45"  # HBA_HUMAN (Hemoglobin alpha)
    print(f"Fetching {protein_id}...")
    summary = client.get_protein_summary(protein_id)

    print(f"\nProtein: {summary['name']}")
    print(f"Organism: {summary['organism']}")
    print(f"GO terms: {len(summary['go_terms'])}")
    print(f"EC numbers: {summary['ec_numbers']}")
    print(f"Pfam domains: {len(summary['pfam_domains'])}")

    if summary['pfam_domains']:
        print("\nPfam domains:")
        for pfam_id, desc in summary['pfam_domains']:
            print(f"- {pfam_id}: {desc}")

    # Test comparison
    print("\n" + "="*70)
    print("Comparing HBA_HUMAN vs HBB_HUMAN...")

    comparison = client.compare_annotations("A0A009HPM0", "Q8DXM9")

    if comparison['comparable']:
        print(f"\nGO similarity: {comparison['go_similarity']:.2f}")
        print(f"Pfam similarity: {comparison['pfam_similarity']:.2f}")
        print(f"Common GO terms: {len(comparison['go_common'])}")
        print(f"common go terms: {comparison['go_common']}")
        print(f"Common Pfam domains: {len(comparison['pfam_common'])}")
        print(f"Has common function: {comparison['has_common_function']}")

    print("\nTest complete!")
