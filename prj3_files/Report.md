# Αναφορά - Εργασία 3

## 1. Ποσοτική Σύγκριση
- Πίνακας Recall@N και QPS
- Ανάλυση speed/accuracy trade-off

## 2. Επιλογή Υπερπαραμέτρων
- LSH: k=6, L=8, w=1.0 
- Hypercube: kproj=12, w=1.5 
- IVF: n_clusters=1000, n_probe=50 
- Neural LSH: m=400, T=50 
## 3. Ορισμός Remote Homolog
Ορίζουμε ως remote homolog μια πρωτεΐνη που:
- Έχει sequence identity <30% (Twilight Zone)
- Βρίσκεται κοντά στο embedding space (μικρή L2)
- Μοιράζεται κοινό Pfam domain ή λειτουργία

## 4. Παραδείγματα Remote Homologs (3-5)
### Παράδειγμα 1: ...
### Παράδειγμα 2: ...

## 5. False Positives for report
- Περιπτώσεις αποτυχίας


python protein_embed.py -i datasets/targets.fasta -o queries
python protein_search.py -d database.npy -q datasets/targets.fasta -method lsh --use-cpp --ground-truth blast_gt.pkl --pfam-map datasets/targets.pfam_map.tsv 
python protein_search.py     -d data/swissprot_vectors.npy     -q data/target_vectors.npy     -o results     -method all     --ground-truth data/blast_results.pkl     --pfam-map data/targets.pfam_map.tsv     --N 50 --lsh-L 5 --lsh-k 4 --lsh-w 2.0 --hc-kproj10 --hc-w 2.0 --hc-max-probes 5 --hc-M 8000 --ivf-n-clusters 1000 --ivf-n-probe 100 --ivf-M 32 --ivf-nbits 8 --nlsh-m 200 --nlsh-k 15 --nlsh-T 30 --nlsh-epochs 15