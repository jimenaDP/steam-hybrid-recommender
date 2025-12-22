[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/e--wBz4e)
# Steam Game Recommender System  
**Content-Based (LDA) · Collaborative Filtering (ItemKNN) · Hybrid (FWLS) · Post-hoc Re-ranking**

This repository contains the implementation and evaluation of a recommender system for Steam games.  
The project explores three main approaches:

- **Content-Based filtering (CB)** using **Latent Dirichlet Allocation (LDA)** over game descriptions.
- **Collaborative Filtering (CF)** using an **ItemKNN** model based on user–item interactions.
- A **Hybrid recommender** combining CB and CF through **Feature-Weighted Linear Stacking (FWLS)**.

In addition, two **post-hoc re-ranking strategies** are applied to the hybrid model to study trade-offs between accuracy, fairness, and diversity:
- **Popularity Penalty**
- **Maximal Marginal Relevance (MMR)**

Evaluation focuses on both **accuracy** (NDCG@20, Recall@20) and **fairness/diversity** metrics.

---

## Data

The following datasets are used:

- `extended_games.csv`  
  Contains rich item metadata (game name, descriptions, reviews, and tags).  
  This dataset is used for the **content-based model**.

- `train_interactions.csv`  
  User–item interactions used to train the models.

- `test_interactions_in.csv`  
  Fold-in interactions used to generate recommendations for evaluation.

## Evaluation Protocol

- For CB and CF models, the provided test set is split into **test-in** and **test-out** to enable faster local experimentation.
- For the **hybrid model and post-hoc strategies**, the official evaluation protocol is followed and results are submitted to **Codabench**.


---

## Key Findings

- The hybrid model is more stable and accurate than CB or CF alone.
- Popularity penalty slightly improves diversity while preserving accuracy.
- MMR*increases diversity more aggressively but at a noticeable cost in accuracy.
- Post-hoc methods act as fine-tuning tools, not replacements for the hybrid model.

---

## Future Work

- Explore alternative re-ranking strategies.
- Investigate switched hybrid models.

---

## References

- Falk, *Content-Based Filtering*, book chapter.  
- Falk, *Taking the Best of All Algorithms: Implementing Hybrid Recommenders*, book chapter.  
- Li et al., *A Survey on Fairness and Diversity in Recommender Systems*.
