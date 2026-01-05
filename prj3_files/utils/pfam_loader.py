from typing import Dict, List, Tuple, Optional
from pathlib import Path


# Common Pfam domain descriptions
PFAM_DESCRIPTIONS = {
    'PF00069': 'Protein kinase domain',
    'PF00005': 'ABC transporter',
    'PF00072': 'Response regulator receiver domain',
    'PF00271': 'Helicase conserved C-terminal domain',
    'PF00364': 'Biotin-requiring enzyme',
    'PF00400': 'WD40 repeat',
    # Add more as needed
}


def load_pfam_mapping(filepath: str) -> Dict[str, dict]:
    mapping = {}
    
    with open(filepath, 'r') as f:
        # Skip header
        header = f.readline().strip().split('\t')
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                accession = parts[0]
                pfam = parts[1]
                length = int(parts[2])
                
                mapping[accession] = {
                    'pfam': pfam,
                    'length': length,
                    'description': PFAM_DESCRIPTIONS.get(pfam, 'Unknown domain')
                }
    
    print(f"Loaded Pfam mapping for {len(mapping)} proteins")
    return mapping


def get_pfam_for_id(protein_id: str, pfam_mapping: Dict[str, dict]) -> Optional[str]:
    # Try direct lookup
    if protein_id in pfam_mapping:
        return pfam_mapping[protein_id]['pfam']
    
    # Try extracting accession from UniProt format
    if '|' in protein_id:
        parts = protein_id.split('|')
        if len(parts) >= 2:
            accession = parts[1]
            if accession in pfam_mapping:
                return pfam_mapping[accession]['pfam']
    
    # Try first part before any separator
    accession = protein_id.split()[0].split('|')[-1]
    if accession in pfam_mapping:
        return pfam_mapping[accession]['pfam']
    
    return None


def check_same_family(
    query_id: str, 
    neighbor_id: str, 
    pfam_mapping: Dict[str, dict]
) -> Tuple[bool, Optional[str], Optional[str]]:

    query_pfam = get_pfam_for_id(query_id, pfam_mapping)
    neighbor_pfam = get_pfam_for_id(neighbor_id, pfam_mapping)
    
    if query_pfam is None or neighbor_pfam is None:
        return False, query_pfam, neighbor_pfam
    
    return query_pfam == neighbor_pfam, query_pfam, neighbor_pfam


def get_family_members(
    pfam_id: str, 
    pfam_mapping: Dict[str, dict]
) -> List[str]:
    return [
        acc for acc, data in pfam_mapping.items()
        if data['pfam'] == pfam_id
    ]


def analyze_pfam_coverage(
    query_ids: List[str],
    database_ids: List[str],
    pfam_mapping: Dict[str, dict]
) -> Dict:

    query_pfams = {}
    for qid in query_ids:
        pfam = get_pfam_for_id(qid, pfam_mapping)
        if pfam:
            if pfam not in query_pfams:
                query_pfams[pfam] = []
            query_pfams[pfam].append(qid)
    
    db_pfams = {}
    for did in database_ids:
        pfam = get_pfam_for_id(did, pfam_mapping)
        if pfam:
            if pfam not in db_pfams:
                db_pfams[pfam] = []
            db_pfams[pfam].append(did)
    
    # Find overlap
    overlap = set(query_pfams.keys()) & set(db_pfams.keys())
    
    return {
        'query_pfams': query_pfams,
        'database_pfams': db_pfams,
        'overlapping_families': list(overlap),
        'num_query_families': len(query_pfams),
        'num_db_families': len(db_pfams),
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mapping = load_pfam_mapping(sys.argv[1])
        
        print("\nPfam families found:")
        families = {}
        for acc, data in mapping.items():
            pfam = data['pfam']
            if pfam not in families:
                families[pfam] = []
            families[pfam].append(acc)
        
        for pfam, members in sorted(families.items()):
            desc = PFAM_DESCRIPTIONS.get(pfam, 'Unknown')
            print(f"  {pfam} ({desc}): {len(members)} proteins")
            for m in members[:3]:
                print(f"    - {m}")
            if len(members) > 3:
                print(f"    ... and {len(members)-3} more")