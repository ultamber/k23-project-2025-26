from Bio import Align

def compute_sequence_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignments = aligner.align(seq1, seq2)
    
    if not alignments:
        return 0.0
    
    alignment = alignments[0]
    aligned_seq1 = alignment.seqA
    aligned_seq2 = alignment.seqB
    
    # Count matches
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    
    # Identity = matches / min(len(seq1), len(seq2))
    identity = 100.0 * matches / min(len(seq1), len(seq2))
    
    return identity