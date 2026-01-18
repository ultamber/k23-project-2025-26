# Αναζήτηση Απομακρυσμένων Ομολόγων Πρωτεϊνών με χρήση ESM-2 Embeddings και ANN Αλγορίθμων

**Εργασία 3 - Χειμερινό Εξάμηνο 2025-2026**
| | |
|---|---|
| **Ονοματεπώνυμο:** | *[Μαρκοπούλου Χριστίνα, Βασταρδής Νικόλαος]* |
| **Αριθμός Μητρώου:** | *[1115201800109, 1115201900020 ]* |


## Οδηγίες Εκτέλεσης

**Παραγωγή Embeddings:**

```bash
python protein_embed.py \
    -i swissprot_50k.fasta \
    -o vectors.npy \
    -model esm2_t6_8M_UR50D
```

**Εκτέλεση BLAST (Ground Truth):**

```bash
# Δημιουργία βάσης
makeblastdb -in swissprot.fasta -dbtype prot -out swissprot_db

# Αναζήτηση
blastp -db swissprot_db -query queries.fasta -outfmt 6 -out blast_results.tsv
```

**Εκτέλεση Αναζήτησης ANN:**

```bash
python protein_search.py \
    -d database.npy \
    -q datasets/targets.fasta \
    -o output \
    -method all \
    --ground-truth blast_gt2.pkl \
    --pfam-map datasets/targets.pfam_map.tsv
```

## Δομή Αρχείων

```
project/
├── blast_gt2.pkl
├── cpp_wrapper.py
├── datasets
│   ├── swissprot_50k.fasta
│   ├── targets.fasta
│   ├── targets_full.pfam_map.tsv
│   └── targets.pfam_map.tsv
├── protein_embed.py
├── protein_search.py
├── README.md
├── Report.md
├── requirements.txt
└── utils
    ├── blast_runner.py
    ├── blast_to_pickle.py
    ├── evaluation.py
    ├── fasta_loader.py
    ├── graph_utils.py
    ├── output_formatter.py
    ├── pfam_loader.py
    ├── protein_fvecs.py
    ├── results_writer.py
    └── uniprot_client.py
```

---

*Τέλος Αναφοράς*
