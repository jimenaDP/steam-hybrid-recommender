# User-Adaptive Hybrid Recommenders and the Diversity–Fairness Trade-off in Post-Hoc Re-ranking

Hybrid recommender system for Steam games. Combines collaborative filtering (CF) and content-based filtering (CB) with Feature-Weighted Linear Stacking (FWLS), then studies three research questions:

- **RQ1** — a user-adaptive α (per-user CF/CB weighting) vs. a single fixed α
- **RQ2** — diversity-oriented post-hoc re-ranking (popularity penalty, MMR)
- **RQ3** — fairness-oriented post-hoc re-ranking (publisher exposure, based on Modani et al.)

Full methodology and results are in the [report](report.pdf).

Developed as the final project of the course *AI Project* at the University of Antwerp (Erasmus exchange, 2025–2026).

## Repository structure

```
.
├── main.ipynb                  # End-to-end pipeline: run this to reproduce all results
├── data_prep.py                # Loading, cleaning, and fold-in/hold-out splitting
├── content_based.py            # CB model: LDA topic model + cosine similarity
├── collaborative_filtering.py  # CF model: ItemKNN on the binary interaction matrix
├── fwls_hybrid.py              # FWLS hybrid: score projection, normalization, fixed-α combination
├── adaptive_hybrid.py          # User-adaptive α hybrid (RQ1)
├── reranking.py                # Popularity penalty and MMR re-ranking (RQ2)
├── reranking_extended.py       # Publisher fairness re-ranking (RQ3)
├── metrics.py                  # NDCG@k, Recall@k, Item Gini@k, ILS@k, per-user metrics
├── submission.py               # Exports recommendation lists to a Codabench-ready CSV
├── AIreport.pdf                  # Project report
└── dataset/                    # Not included — see Data setup below
```

## Requirements

```
python >= 3.9
pandas
numpy
scipy
scikit-learn
gensim
nltk
matplotlib
```

Install with:

```bash
pip install pandas numpy scipy scikit-learn gensim nltk matplotlib
```

`content_based.py` downloads the NLTK English stopword list automatically on first run if it isn't already present.

## Data setup

Place the following files in a `dataset/` folder at the repository root:

```
dataset/
├── extended_games.csv
├── train_interactions.csv
└── test_interactions_in.csv
```

These come from the course's Steam dataset (Codabench). `data_prep.py` reads them, cleans them, and writes cleaned copies back to `dataset/` (`extended_games_clean.csv`, `train_data_clean.csv`).

## How to run

Open `main.ipynb` and run the cells top to bottom. It is organized into sections that mirror the pipeline:

1. **Preprocessing** — load and clean the raw data, split test interactions into fold-in / hold-out
2. **Content-Based Model** — train LDA, generate and evaluate CB recommendations
3. **Collaborative Filtering** — build the ItemKNN model, generate and evaluate CF recommendations
4. **FWLS Hybrid Recommender** — project and normalize CF/CB scores, sweep α, select the best fixed α
5. **User-Adaptive Hybrid Recommender** — sweep threshold T, compare against the fixed hybrid (RQ1)
6. **Reranking Methods** — sweep γ (popularity penalty), λ (MMR), and β (fairness re-ranking), compare all three (RQ2, RQ3)
7. **Results** — select the best parameter per method, generate final recommendation lists, plot per-user metric distributions


## Outputs

Running the notebook produces one submission file per method, ready to upload to Codabench:

- `submission_fwls.csv` — fixed-α hybrid
- `submission_adaptive.csv` — user-adaptive hybrid
- `submission_mmr.csv` — MMR re-ranking
- `submission_fairness.csv` — fairness re-ranking

## Author

Jimena de Prado González
