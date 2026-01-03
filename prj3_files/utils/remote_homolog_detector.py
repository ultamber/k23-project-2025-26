#!/usr/bin/env python3
"""
Remote Homolog Detector

Identifies potential remote homolog pairs:
- Low BLAST sequence identity (< 30%, Twilight Zone)
- Small L2 distance in embedding space
- Common biological function (validated via UniProt)

"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from utils.uniprot_client import UniProtClient
from utils.evaluation import compute_sequence_identity


class RemoteHomologDetector:
    """Detect and validate remote homolog candidates."""
    
    def __init__(
        self,
        identity_threshold: float = 30.0,
        l2_threshold: float = 5.0,
        min_go_similarity: float = 0.3,
        min_pfam_similarity: float = 0.3,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize detector.
        
        Args:
            identity_threshold: Max BLAST identity for remote homologs (%)
            l2_threshold: Max L2 distance threshold
            min_go_similarity: Min GO term overlap for validation
            min_pfam_similarity: Min Pfam domain overlap for validation
            cache_dir: UniProt cache directory
        """
        self.identity_threshold = identity_threshold
        self.l2_threshold = l2_threshold
        self.min_go_similarity = min_go_similarity
        self.min_pfam_similarity = min_pfam_similarity
        
        self.uniprot = UniProtClient(cache_dir=cache_dir)
    
    def is_remote_homolog_candidate(
        self,
        l2_distance: float,
        blast_identity: Optional[float],
        in_blast_top_n: Optional[bool]
    ) -> Tuple[bool, str]:
        """
        Check if a neighbor is a remote homolog candidate.
        
        Args:
            l2_distance: L2 distance in embedding space
            blast_identity: BLAST sequence identity (%)
            in_blast_top_n: Whether neighbor is in BLAST Top-N
            
        Returns:
            (is_candidate, reason)
        """
        # Must be close in embedding space
        if l2_distance > self.l2_threshold:
            return False, f"L2 distance too high ({l2_distance:.2f} > {self.l2_threshold})"
        
        # If no BLAST data, consider it based on L2 only
        if blast_identity is None:
            return True, "Close in embedding space (no BLAST data)"
        
        # Remote homolog: low BLAST identity but close in embedding space
        if blast_identity < self.identity_threshold:
            return True, f"Low BLAST identity ({blast_identity:.1f}% < {self.identity_threshold}%) + close embedding"
        
        # High identity - not a remote homolog
        return False, f"High BLAST identity ({blast_identity:.1f}%)"
    
    def validate_remote_homolog(
        self,
        protein_id1: str,
        protein_id2: str,
        l2_distance: float,
        blast_identity: Optional[float]
    ) -> Dict:
        """
        Validate remote homolog with biological evidence.
        
        Args:
            protein_id1: Query protein ID
            protein_id2: Candidate neighbor ID
            l2_distance: L2 distance
            blast_identity: BLAST identity (%)
            
        Returns:
            Validation dict with biological evidence
        """
        # Get annotations
        comparison = self.uniprot.compare_annotations(protein_id1, protein_id2)
        
        if not comparison['comparable']:
            return {
                'validated': False,
                'reason': comparison['reason'],
                'confidence': 'unknown'
            }
        
        # Check biological similarity
        has_common_pfam = comparison['pfam_similarity'] >= self.min_pfam_similarity
        has_common_go = comparison['go_similarity'] >= self.min_go_similarity
        has_common_ec = comparison['ec_match']
        
        # Evidence of homology
        evidence = []
        if has_common_pfam:
            evidence.append(f"Common Pfam domains ({len(comparison['pfam_common'])})")
        if has_common_go:
            evidence.append(f"Similar GO terms (sim={comparison['go_similarity']:.2f})")
        if has_common_ec:
            evidence.append(f"Same EC class ({comparison['ec_common']})")
        
        # Determine confidence
        if has_common_pfam or has_common_ec:
            confidence = 'high'
            validated = True
        elif has_common_go:
            confidence = 'medium'
            validated = True
        else:
            confidence = 'low'
            validated = False
        
        return {
            'validated': validated,
            'confidence': confidence,
            'evidence': evidence,
            'comparison': comparison,
            'is_remote_homolog': validated and (blast_identity is None or blast_identity < self.identity_threshold)
        }
    
    def detect_remote_homologs(
        self,
        query_id: str,
        neighbors: List[Tuple[int, float]],
        database_ids: List[str],
        database_seqs: Optional[List[str]] = None,
        query_seq: Optional[str] = None,
        blast_results: Optional[Dict] = None,
        query_idx: int = 0,
        N: int = 50,
        max_candidates: int = 10
    ) -> List[Dict]:
        """
        Detect remote homologs for a query protein.
        
        Args:
            query_id: Query protein ID
            neighbors: List of (neighbor_idx, l2_distance)
            database_ids: Database protein IDs
            database_seqs: Database sequences
            query_seq: Query sequence
            blast_results: BLAST ground truth
            query_idx: Query index
            N: Top-N for BLAST comparison
            max_candidates: Max candidates to validate
            
        Returns:
            List of remote homolog dicts
        """
        candidates = []
        
        for neighbor_idx, l2_distance in neighbors[:max_candidates]:
            # Get neighbor ID
            if neighbor_idx >= len(database_ids):
                continue
            
            neighbor_id = database_ids[neighbor_idx]
            
            # Compute BLAST identity if sequences available
            blast_identity = None
            if query_seq and database_seqs and neighbor_idx < len(database_seqs):
                blast_identity = compute_sequence_identity(
                    query_seq,
                    database_seqs[neighbor_idx]
                )
            
            # Check if in BLAST Top-N
            in_blast_top_n = None
            if blast_results and 'blast_results_indices' in blast_results:
                if query_idx in blast_results['blast_results_indices']:
                    blast_top_n = [idx for idx, _, _ in blast_results['blast_results_indices'][query_idx][:N]]
                    in_blast_top_n = neighbor_idx in blast_top_n
            
            # Check if candidate
            is_candidate, reason = self.is_remote_homolog_candidate(
                l2_distance,
                blast_identity,
                in_blast_top_n
            )
            
            if not is_candidate:
                continue
            
            # Validate with biological evidence
            validation = self.validate_remote_homolog(
                query_id,
                neighbor_id,
                l2_distance,
                blast_identity
            )
            
            candidates.append({
                'query_id': query_id,
                'neighbor_id': neighbor_id,
                'neighbor_idx': neighbor_idx,
                'l2_distance': l2_distance,
                'blast_identity': blast_identity,
                'in_blast_top_n': in_blast_top_n,
                'is_candidate': is_candidate,
                'candidate_reason': reason,
                **validation
            })
        
        # Sort by confidence and L2 distance
        confidence_order = {'high': 0, 'medium': 1, 'low': 2, 'unknown': 3}
        candidates.sort(key=lambda x: (
            confidence_order.get(x.get('confidence', 'unknown'), 3),
            x['l2_distance']
        ))
        
        return candidates
    
    def generate_bio_comment(
        self,
        l2_distance: float,
        blast_identity: Optional[float],
        in_blast_top_n: Optional[bool],
        validation: Optional[Dict] = None
    ) -> str:
        """
        Generate biological comment for output table.
        
        Args:
            l2_distance: L2 distance
            blast_identity: BLAST identity (%)
            in_blast_top_n: In BLAST Top-N?
            validation: Validation dict
            
        Returns:
            Bio comment string
        """
        # No validation data
        if not validation:
            if blast_identity is not None and blast_identity < self.identity_threshold:
                return "Remote homolog?"
            return "--"
        
        # Validated remote homolog
        if validation.get('is_remote_homolog'):
            evidence = validation.get('evidence', [])
            if evidence:
                # Use first piece of evidence
                return f"Remote homolog: {evidence[0]}"
            return "Remote homolog (validated)"
        
        # Not validated
        if validation.get('validated'):
            return "Similar function (high identity)"
        
        # False positive
        if blast_identity is not None and blast_identity < self.identity_threshold:
            return "Possible false positive"
        
        return "--"
    
    def generate_case_study(
        self,
        remote_homolog: Dict,
        detailed: bool = True
    ) -> str:
        """
        Generate case study text for a remote homolog.
        
        Args:
            remote_homolog: Remote homolog dict from detect_remote_homologs
            detailed: Include detailed comparison
            
        Returns:
            Formatted case study text
        """
        lines = []
        
        lines.append(f"\n{'='*70}")
        lines.append("REMOTE HOMOLOG CASE STUDY")
        lines.append(f"{'='*70}\n")
        
        # Basic info
        lines.append(f"Query:    {remote_homolog['query_id']}")
        lines.append(f"Neighbor: {remote_homolog['neighbor_id']}")
        lines.append(f"L2 Distance: {remote_homolog['l2_distance']:.4f}")
        
        if remote_homolog['blast_identity'] is not None:
            lines.append(f"BLAST Identity: {remote_homolog['blast_identity']:.1f}% (< 30% = Twilight Zone)")
        else:
            lines.append(f"BLAST Identity: N/A")
        
        lines.append(f"In BLAST Top-N: {remote_homolog.get('in_blast_top_n', '?')}")
        lines.append(f"Validation: {remote_homolog.get('confidence', 'unknown').upper()}")
        lines.append("")
        
        # Evidence
        evidence = remote_homolog.get('evidence', [])
        if evidence:
            lines.append("Biological Evidence:")
            for ev in evidence:
                lines.append(f"• {ev}")
            lines.append("")
        
        # Detailed comparison
        if detailed and 'comparison' in remote_homolog:
            comp = remote_homolog['comparison']
            
            if comp.get('comparable'):
                lines.append("Detailed Comparison:")
                lines.append("")
                
                # Protein names
                p1 = comp['protein1']
                p2 = comp['protein2']
                
                lines.append(f"Query:    {p1.get('name', 'Unknown')}")
                lines.append(f"          Organism: {p1.get('organism', 'Unknown')}")
                lines.append(f"Neighbor: {p2.get('name', 'Unknown')}")
                lines.append(f"          Organism: {p2.get('organism', 'Unknown')}")
                lines.append("")
                
                # Pfam domains
                if comp['pfam_common']:
                    lines.append(f"Common Pfam Domains ({len(comp['pfam_common'])}):")
                    for pfam_id in comp['pfam_common'][:5]:  # Show top 5
                        # Find description
                        for pid, desc in p1.get('pfam_domains', []):
                            if pid == pfam_id:
                                lines.append(f"  - {pfam_id}: {desc}")
                                break
                    lines.append("")
                
                # EC numbers
                if comp['ec_common']:
                    lines.append(f"Common EC Numbers: {', '.join(comp['ec_common'])}")
                    lines.append("")
                
                # GO terms
                if comp['go_common']:
                    lines.append(f"Common GO Terms ({len(comp['go_common'])} of {len(comp['go_common']) + len(comp['go_only1']) + len(comp['go_only2'])}):")
                    # Show a few examples
                    for go_id in list(comp['go_common'])[:3]:
                        # Find term description
                        for gid, aspect, term in p1.get('go_terms', []):
                            if gid == go_id:
                                lines.append(f"  - {go_id} ({aspect}): {term}")
                                break
                    lines.append("")
                
                # Similarity scores
                lines.append(f"GO Similarity: {comp.get('go_similarity', 0):.2f}")
                lines.append(f"Pfam Similarity: {comp.get('pfam_similarity', 0):.2f}")
                lines.append("")
        
        # Conclusion
        lines.append("Conclusion:")
        if remote_homolog.get('is_remote_homolog'):
            lines.append("  CONFIRMED REMOTE HOMOLOG")
            lines.append("    - Low sequence identity but similar biological function")
            lines.append("    - ESM-2 embeddings successfully captured functional similarity")
        else:
            lines.append("  ✗ NOT CONFIRMED")
            lines.append("    - Insufficient biological evidence")
        
        lines.append(f"\n{'='*70}\n")
        
        return "\n".join(lines)


if __name__ == '__main__':
    """Test remote homolog detector."""
    
    print("Testing Remote Homolog Detector...\n")
    
    # Create detector
    detector = RemoteHomologDetector(
        identity_threshold=30.0,
        l2_threshold=5.0,
        cache_dir='uniprot_cache'
    )
    
    # Test with hemoglobin proteins
    # HBA_HUMAN (P69905) vs HBB_HUMAN (P68871) - known to have low sequence identity but similar function
    
    print("Testing: HBA_HUMAN vs HBB_HUMAN")
    print("(Known remote homolog pair - same function, different sequence)\n")
    
    # Simulate detection
    candidates = [
        (10, 0.15),  # HBB_HUMAN - low distance
        (25, 0.35),  # Some other protein
    ]
    
    database_ids = ['P12345'] * 9 + ['P68871'] + ['Q99999'] * 15 + ['P02008'] + ['X00000'] * 9
    
    results = detector.detect_remote_homologs(
        query_id='P69905',
        neighbors=candidates,
        database_ids=database_ids,
        query_idx=0,
        max_candidates=2
    )
    
    print(f"Found {len(results)} candidates\n")
    
    for i, result in enumerate(results, 1):
        print(f"Candidate {i}:")
        print(f"Neighbor: {result['neighbor_id']}")
        print(f"L2 Distance: {result['l2_distance']:.4f}")
        print(f"Confidence: {result.get('confidence', 'unknown')}")
        print(f"Is Remote Homolog: {result.get('is_remote_homolog', False)}")
        print(f"Evidence: {result.get('evidence', [])}")
        print()
    
    # Generate case study
    if results and results[0].get('is_remote_homolog'):
        print("\n" + "="*70)
        print("CASE STUDY:")
        print("="*70)
        case_study = detector.generate_case_study(results[0])
        print(case_study)
    
    print("Test complete!")