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
}


def load_pfam_mapping(filepath: str) -> Dict[str, dict]:
    """
    Load Pfam domain mapping from a tab-separated file.

    Expected file format: accession<TAB>length<TAB>pfam1;pfam2;...
    Creates a mapping from protein accessions to their Pfam domains and metadata.
    """
    mapping = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        print("Header:", header)

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()   # IMPORTANT
            if len(parts) < 3:
                continue

            accession = parts[0]
            length = int(parts[1])

            # pfam like "PF17652;PF03639;" -> list
            pfams = [p for p in parts[2].split(";") if p]

            mapping[accession] = {
                "length": length,
                "pfams": pfams,
                "pfam": pfams[0] if pfams else None,
            }

    print(f"Loaded Pfam mapping for {len(mapping)} proteins")
    return mapping



def get_pfams_for_id(protein_id: str, pfam_mapping: Dict[str, dict]) -> List[str]:
    """
    Get all Pfam domains for a protein ID, handling various ID formats.
    Supports direct accession lookup, UniProt sp|ACC|NAME format, and plain accessions.
    Tries multiple lookup strategies to find domain information.
    """
    def lookup(key: str) -> List[str]:
        if key in pfam_mapping:
            return pfam_mapping[key].get("pfams", []) or []
        return []

    if protein_id in pfam_mapping:
        return lookup(protein_id)

    token = protein_id.split()[0]  # remove extra stuff

    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 3:
            acc = parts[1]
            name = parts[2]
            return lookup(acc) or lookup(name)

    return lookup(token)

def _get_pfams_for_id(protein_id: str, pfam_mapping: Dict) -> List[str]:
    """
    Helper function to get Pfam domains for a protein ID (internal use).
    Similar to get_pfams_for_id but with different fallback logic.
    """
    def lookup(key: str) -> List[str]:
        if key in pfam_mapping:
            pfams = pfam_mapping[key].get("pfams")
            if pfams is None:
                # fallback if mapping only has single 'pfam'
                p = pfam_mapping[key].get("pfam")
                return [p] if p else []
            return [p for p in pfams if p]
        return []

    token = protein_id.split()[0]

    # direct
    pf = lookup(protein_id)
    if pf:
        return pf

    # UniProt sp|ACC|NAME
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 3:
            acc = parts[1]
            name = parts[2]
            return lookup(acc) or lookup(name) or []

    # plain accession
    return lookup(token) or []

def _get_pfam_for_id(protein_id: str, pfam_mapping: Dict) -> Optional[str]:
    """
    Helper function to get the primary Pfam domain for a protein ID (internal use).
    Returns the first Pfam domain from the list, or None if no domains found.
    """
    pfams = _get_pfams_for_id(protein_id, pfam_mapping)
    return pfams[0] if pfams else None

def get_pfam_for_id(protein_id: str, pfam_mapping: Dict[str, dict]) -> Optional[str]:
    """
    Get the primary Pfam domain for a protein ID.
    Returns the first Pfam domain associated with the protein, handling various ID formats.
    """
    if protein_id in pfam_mapping:
        return pfam_mapping[protein_id]['pfam']

    token = protein_id.split()[0]

    # UniProt: sp|ACCESSION|NAME
    if '|' in token:
        parts = token.split('|')
        if len(parts) >= 3:
            acc = parts[1]
            name = parts[2]
            if acc in pfam_mapping:
                return pfam_mapping[acc]['pfam']
            if name in pfam_mapping:
                return pfam_mapping[name]['pfam']

    # plain accession
    if token in pfam_mapping:
        return pfam_mapping[token]['pfam']

    return None

def check_same_family(query_id: str, neighbor_id: str, pfam_mapping: Dict[str, dict]):
    """
    Check if two proteins belong to the same Pfam domain family.
    Determines if query and neighbor proteins share any Pfam domains.
    """
    q = set(get_pfams_for_id(query_id, pfam_mapping))
    n = set(get_pfams_for_id(neighbor_id, pfam_mapping))

    if not q or not n:
        return False, list(q) or None, list(n) or None

    return len(q & n) > 0, sorted(q), sorted(n)



def get_family_members(pfam_id: str, pfam_mapping: Dict[str, dict]) -> List[str]:
    """
    Get all proteins that contain a specific Pfam domain.
    Finds all protein accessions that have the given Pfam domain.
    """
    out = []
    for acc, data in pfam_mapping.items():
        pfams = data.get("pfams", []) or []
        if pfam_id in pfams:
            out.append(acc)
    return out


def analyze_pfam_coverage(query_ids: List[str], database_ids: List[str], pfam_mapping: Dict[str, dict]) -> Dict:
    """
    Analyze Pfam domain coverage across query and database sets.
    Computes statistics about which Pfam families are present in queries vs database,
    and identifies overlapping families.
    """
    query_pfams: Dict[str, List[str]] = {}
    for qid in query_ids:
        for pf in get_pfams_for_id(qid, pfam_mapping):
            query_pfams.setdefault(pf, []).append(qid)

    db_pfams: Dict[str, List[str]] = {}
    for did in database_ids:
        for pf in get_pfams_for_id(did, pfam_mapping):
            db_pfams.setdefault(pf, []).append(did)

    overlap = set(query_pfams.keys()) & set(db_pfams.keys())

    return {
        "query_pfams": query_pfams,
        "database_pfams": db_pfams,
        "overlapping_families": sorted(overlap),
        "num_query_families": len(query_pfams),
        "num_db_families": len(db_pfams),
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
                print(f"- {m}")
            if len(members) > 3:
                print(f"... and {len(members)-3} more")