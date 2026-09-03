#Imports
import numpy as np
from collections import Counter

"""Recommendations are passed as a list, where each position holds one user's
ranked list of recommended item_ids. This list lines up with the separate
user_ids list, so that recommendations[i] belongs to user_ids[i]"""

def _ground_truth_dict(test_out_df):
    """Convert hold-out DataFrame into {user_id: set(item_ids)}"""
    return test_out_df.groupby("user_id")["item_id"].apply(set).to_dict()


def ndcg_at_k(recommendations, user_ids, test_out_df, k=20):
    """Normalized Discounted Cumulative Gain @ k, averaged over users."""
    gt = _ground_truth_dict(test_out_df)
    scores = []

    for user_id, rec_list in zip(user_ids, recommendations):
        if user_id not in gt:
            continue
        relevant = gt[user_id]
        rec_list = rec_list[:k]

        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_list) if item in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(scores) if scores else 0.0


def recall_at_k(recommendations, user_ids, test_out_df, k=20):
    """Recall@k,"""
    gt = _ground_truth_dict(test_out_df)
    scores = []

    for user_id, rec_list in zip(user_ids, recommendations):
        if user_id not in gt:
            continue
        relevant = gt[user_id]
        if len(relevant) == 0:
            continue
        rec_list = rec_list[:k]

        hits = sum(1 for item in rec_list if item in relevant)
        scores.append(hits / len(relevant))

    return np.mean(scores) if scores else 0.0


def item_gini(recommendations, k=20):
    """Item Gini@k"""
    counts = Counter()
    for rec_list in recommendations:
        for item in rec_list[:k]:
            counts[item] += 1

    if not counts:
        return 0.0

    values = np.sort(np.array(list(counts.values())))
    n = len(values)
    cum_values = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n + 1)) * values) - (n + 1) * cum_values[-1]) / (n * cum_values[-1])

    return gini


def intra_list_similarity(recommendations, similarity_matrix, id_index, k=20):
    """ILS@k"""
    ils_scores = []

    for rec_list in recommendations:
        rec_list = rec_list[:k]
        idx = [id_index[item] for item in rec_list if item in id_index]
        if len(idx) < 2:
            continue

        sims = similarity_matrix[np.ix_(idx, idx)]
        n = len(idx)
        avg_sim = (sims.sum() - n) / (n * (n - 1))  #exclude diagonal (self-similarity = 1)
        ils_scores.append(avg_sim)

    return np.mean(ils_scores) if ils_scores else 0.0

def select_best_param(df, baseline_param, metric_col, higher_is_better, ndcg_budget=0.75):
    """Selects the parameter that optimizes a metric, within an NDCG budget relative to baseline"""
    baseline_ndcg = df.loc[df["param"] == baseline_param, "ndcg"].values[0]
    acceptable = df[df["ndcg"] >= baseline_ndcg * ndcg_budget]
    pool = acceptable if len(acceptable) > 0 else df

    if higher_is_better:
        best_param = pool.loc[pool[metric_col].idxmax(), "param"]
    else:
        best_param = pool.loc[pool[metric_col].idxmin(), "param"]

    row = df.loc[df["param"] == best_param].iloc[0]
    print(f"Chosen param={best_param} | ndcg={row['ndcg']:.4f} | recall={row['recall']:.4f} | "
          f"gini={row['gini']:.4f} | ils={row['ils']:.4f}"
          + (f" | fairness={row['fairness']:.4f}" if "fairness" in row else ""))
    return best_param

def per_user_ndcg(recommendations, user_ids, test_out_df, k=20):
    """Per-user NDCG values,for boxplotting."""
    gt = _ground_truth_dict(test_out_df)
    scores = []
    for user_id, rec_list in zip(user_ids, recommendations):
        if user_id not in gt:
            continue
        relevant = gt[user_id]
        rec_list = rec_list[:k]
        dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(rec_list) if item in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return scores


def per_user_ils(recommendations, similarity_matrix, id_index, k=20):
    """Per-user ILS values, for boxplotting."""
    ils_scores = []
    for rec_list in recommendations:
        rec_list = rec_list[:k]
        idx = [id_index[item] for item in rec_list if item in id_index]
        if len(idx) < 2:
            continue
        sims = similarity_matrix[np.ix_(idx, idx)]
        n = len(idx)
        avg_sim = (sims.sum() - n) / (n * (n - 1))
        ils_scores.append(avg_sim)
    return ils_scores

