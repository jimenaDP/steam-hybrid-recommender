#Imports
import numpy as np
from fwls_hybrid import recommend_hybrid
from metrics import ndcg_at_k, recall_at_k, item_gini

"""
User-adaptive hybrid: instead of one global alpha for every user, alpha 
is chosen per user based on how much interaction history
they have. Users with fewer than T interactions get more weight
on CB, which does not need much history to produce decent recommendation.
Users with T or more interactions get more weight on CF.
"""

def count_interactions(fold_in_df, user_ids):
    """Number of fold-in interactions per user"""
    counts = fold_in_df.groupby("user_id").size()
    return counts.reindex(user_ids, fill_value=0).values.astype(int)


def switched_alpha(interaction_counts, T, alpha_low=0.3, alpha_high=0.9):
    """
    Users with fewer than T interactions get alpha_low. Users with T or
    more interactions get alpha_high
    """
    alpha_per_user = np.where(interaction_counts < T, alpha_low, alpha_high)

    #sanity check: every value must be exactly alpha_low or alpha_high
    valid_values = {alpha_low, alpha_high}
    assert set(np.unique(alpha_per_user)).issubset(valid_values), \
        f"unexpected alpha values found, expected only {valid_values}"

    return alpha_per_user


def combine_scores_adaptive(cf_scores_norm, cb_scores_norm, alpha_per_user):
    """
    Score_hyb(u,i) = alpha_u * CF(u,i) + (1 - alpha_u) * CB(u,i)

    Same FWLS formula as hybrid.combine_scores, but alpha is a per-user
    array instead of one scalar for everyone
    """
    alpha_col = alpha_per_user.reshape(-1, 1)  #broadcast alpha_u across each user's row
    return alpha_col * cf_scores_norm + (1 - alpha_col) * cb_scores_norm


def check_threshold(cf_scores_norm, cb_scores_norm, user_item_matrix_cb, cb_item_ids,
                     cb_user_ids, interaction_counts, test_out_df, topn=20,
                     thresholds=(1, 2, 3, 5, 8, 13, 21, 34), alpha_low=0.3, alpha_high=0.9):
    results = []
    for T in thresholds:
        alpha_per_user = switched_alpha(interaction_counts, T, alpha_low, alpha_high)
        scores_adaptive = combine_scores_adaptive(cf_scores_norm, cb_scores_norm, alpha_per_user)
        recs = recommend_hybrid(scores_adaptive, user_item_matrix_cb, cb_item_ids, topn=topn)

        ndcg = ndcg_at_k(recs, cb_user_ids, test_out_df, k=topn)
        recall = recall_at_k(recs, cb_user_ids, test_out_df, k=topn)
        gini = item_gini(recs, k=topn)

        print(f"T={T:>3} | ndcg@{topn}={ndcg:.4f} | recall@{topn}={recall:.4f} | gini@{topn}={gini:.4f}")
        results.append({"T": T, "ndcg": ndcg, "recall": recall, "gini": gini})

    return results
