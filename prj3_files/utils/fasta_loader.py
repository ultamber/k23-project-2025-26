from typing import List, Tuple, Dict

def load_fasta(filepath: str) -> List[Tuple[str, str]]:

    sequences = []
    current_id = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))

                # Parse new header
                header = line[1:]

                # Extract ID
                protein_id = header.split()[0] if ' ' in header else header

                current_id = protein_id
                current_seq = []
            else:
                # Accumulate sequence
                current_seq.append(line.replace(' ', ''))

        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))

    print(f"Loaded {len(sequences)} sequences from {filepath}")
    return sequences


def save_fasta(sequences: List[Tuple[str, str]], filepath: str):

    with open(filepath, 'w') as f:
        for protein_id, sequence in sequences:
            f.write(f">{protein_id}\n")

            # Write sequence in lines of 60 characters
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')

    print(f"Saved {len(sequences)} sequences to {filepath}")


def parse_uniprot_id(header: str) -> Dict[str, str]:

    parts = header.split('|')

    result = {
        'full_id': header.split()[0],
        'database': None,
        'accession': None,
        'entry_name': None,
        'description': None,
        'organism': None
    }

    if len(parts) >= 3:
        result['database'] = parts[0]
        result['accession'] = parts[1]

        rest = parts[2]
        if ' ' in rest:
            result['entry_name'] = rest.split()[0]
            result['description'] = ' '.join(rest.split()[1:])
        else:
            result['entry_name'] = rest

    if 'OS=' in header:
        os_start = header.index('OS=') + 3
        os_end = header.find(' OX=', os_start) if ' OX=' in header else len(header)
        result['organism'] = header[os_start:os_end].strip()

    return result


def get_accession(protein_id: str) -> str:

    if '|' in protein_id:
        parts = protein_id.split('|')
        if len(parts) >= 2:
            return parts[1]

    # If no pipes, assume it's already an accession
    return protein_id.split()[0]


if __name__ == '__main__':

    import sys

    if len(sys.argv) > 1:
        sequences = load_fasta(sys.argv[1])
        print(f"\nFirst sequence:")
        print(f"ID: {sequences[0][0]}")
        print(f"Length: {len(sequences[0][1])}")
        print(f"Sequence (first 60): {sequences[0][1][:60]}")

        info = parse_uniprot_id(sequences[0][0])
        print(f"\nParsed info:")
        for key, value in info.items():
            print(f"{key}: {value}")
