#Imports
import numpy as np
from metrics import ndcg_at_k, recall_at_k, item_gini

"""
Hybrid recommender combining CB and CF scores via Feature-Weighted Linear
Stacking (FWLS): score_hyb = alpha * score_cf + (1 - alpha) * score_cb.
CF is trained on a global user/item space; CB is trained on a smaller item 
space (test_in users x extended_games items). CF scores
must be projected into CB space before combining.
"""

def project_cf_to_cb_space(cf_scores, all_users, all_items, cb_user_ids, cb_item_ids):
    """Reduce CF's scores from the full global user/item space down to the smaller
    CB user/item space, so both models' scores can be combined"""

    print(f"Starting CF->CB projection: reducing {len(all_users)} users x {len(all_items)} items "
        f"(CF space) down to {len(cb_user_ids)} users x {len(cb_item_ids)} items (CB space)...")
    user2row_global = {u: i for i, u in enumerate(all_users)}
    item2col_global = {it: i for i, it in enumerate(all_items)}

    user_rows = [user2row_global[u] for u in cb_user_ids if u in user2row_global]
    valid_users = [u for u in cb_user_ids if u in user2row_global]

    cb_item_cols = [item2col_global.get(it, None) for it in cb_item_ids] #return None if it isn't in CF's matrix
    valid_mask = np.array([c is not None for c in cb_item_cols])
    valid_cols = [c for c in cb_item_cols if c is not None]

    projected = np.zeros((len(cb_user_ids), len(cb_item_ids)))
    sub_scores = cf_scores[np.ix_(user_rows, valid_cols)] #only items that exist in CF and CB matrices
    projected[:, valid_mask] = sub_scores

    n_missing_items = (~valid_mask).sum()
    n_missing_users = len(cb_user_ids) - len(valid_users)
    print(f"CF->CB projection COMPLETED: {n_missing_items} CB items not found in global CF space, "
          f"{n_missing_users} CB users not found in global CF space")

    return projected


def minmax_normalize(scores):
    """Per-user min-max normalization"""
    normalized = np.zeros_like(scores)
    for u in range(scores.shape[0]):
        row = scores[u]
        nonzero = row[row != 0]
        if len(nonzero) == 0:
            continue
        row_min, row_max = nonzero.min(), nonzero.max()
        if row_max == row_min:
            normalized[u] = np.where(row != 0, 1.0, 0.0) #To avoid dividing by 0
        else:
            normalized[u] = np.where(row != 0, (row - row_min) / (row_max - row_min), 0.0)

    #sanity check: normalized scores must stay within [0, 1]
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0, \
        f"normalized scores out of range: [{normalized.min():.4f}, {normalized.max():.4f}]"

    return normalized


def combine_scores(cf_scores_norm, cb_scores_norm, alpha):
    """Score_hyb = alpha * CF + (1 - alpha) * CB."""
    return alpha * cf_scores_norm + (1 - alpha) * cb_scores_norm


def recommend_hybrid(hybrid_scores, user_item_matrix_cb, cb_item_ids, topn=20):
    """Discard already-seen CB items, rank, and return top-N per user."""
    scores = hybrid_scores.copy()
    seen_mask = user_item_matrix_cb.toarray().astype(bool) #already seen games
    scores[seen_mask] = -np.inf

    recommendations = []
    for u in range(scores.shape[0]):
        user_scores = scores[u]
        k = min(topn, np.sum(user_scores > -np.inf))
        if k == 0:
            recommendations.append([]) #empty list if the user has seen every game already
            continue
        top_cols = np.argpartition(-user_scores, k - 1)[:k]
        top_cols = top_cols[np.argsort(-user_scores[top_cols])]
        recommendations.append([cb_item_ids[c] for c in top_cols])

    return recommendations


def check_alpha(cf_scores_norm, cb_scores_norm, user_item_matrix_cb, cb_item_ids,
                 cb_user_ids, test_out_df, topn=20, alphas=np.arange(0, 1.05, 0.1)):
    """Evaluate NDCG/Recall/Gini across a range of alpha values."""
    results = []
    for alpha in alphas:
        hybrid_scores = combine_scores(cf_scores_norm, cb_scores_norm, alpha)
        recs = recommend_hybrid(hybrid_scores, user_item_matrix_cb, cb_item_ids, topn=topn)

        ndcg = ndcg_at_k(recs, cb_user_ids, test_out_df, k=topn)
        recall = recall_at_k(recs, cb_user_ids, test_out_df, k=topn)
        gini = item_gini(recs, k=topn)

        print(f"alpha={alpha:.2f} | ndcg@{topn}={ndcg:.4f} | recall@{topn}={recall:.4f} | gini@{topn}={gini:.4f}")
        results.append({"alpha": alpha, "ndcg": ndcg, "recall": recall, "gini": gini})

    return results