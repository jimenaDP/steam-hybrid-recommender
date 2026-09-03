import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon
from collections import Counter
from metrics import ndcg_at_k, recall_at_k, item_gini, intra_list_similarity
from fwls_hybrid import recommend_hybrid

def publisher_capital(extended_games_cleaned, alpha=0.75):
    """
    Creative Capital per publisher. Combines total games, positive reviews, negative
    reviews, and recommendations using PCA: the first principal component
    gives a data-driven composite score, instead of manually chosen weights.
    """
    pub_stats = extended_games_cleaned.groupby("publisher").agg( #group by publisher
        n_games=("item_id", "count"),
        positive=("positive", "sum"),
        negative=("negative", "sum"),
        recommendations=("recommendations", "sum"),
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pub_stats)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(scaled).flatten()

    # pca.fit_transform() computes a weighted sum of the 4 scaled attributes for
    # every publisher (weight_games*games + weight_positive*positive + ...),
    # using PCA-derived weights instead of manually chosen ones. 
    # This is the paper's Cu(t) = wp*np + wa*na + wv*nv formula — just
    # computed via PCA

    if np.corrcoef(pc1, pub_stats["recommendations"])[0, 1] < 0: #check pca's direction
        pc1 = -pc1

    capital = pc1 - pc1.min() + 1e-6 #just above 0, to avoid getting (negative number) ** alpha

    explained_var = pca.explained_variance_ratio_[0]
    print(f"PCA - first component explains {explained_var:.1%} of variance "
          f"across {len(pub_stats)} publishers")
    print(f"Weights: games={pca.components_[0][0]:.3f}, "
          f"positive={pca.components_[0][1]:.3f}, negative={pca.components_[0][2]:.3f}, "
          f"recommendations={pca.components_[0][3]:.3f}")

    sublinear_capital = capital ** alpha
    desired_exposure = pd.Series(
        sublinear_capital / sublinear_capital.sum(), index=pub_stats.index
    )

    return desired_exposure


def actual_exposure(recommendations, item_to_publisher, k=20):
    """Actual exposure share per publisher, based on recommendation frequency."""
    counts = Counter()
    for rec_list in recommendations:
        for item in rec_list[:k]:
            publisher = item_to_publisher.get(item)
            if publisher is not None:
                counts[publisher] += 1

    total = sum(counts.values())
    if total == 0:
        return pd.Series(dtype=float)

    return pd.Series(counts) / total


def publisher_fairness_score(desired_exposure, actual_exposure):
    """Fairness = 1 / JSD(desired || actual). 
    Higher -> fairer."""
    all_publishers = desired_exposure.index.union(actual_exposure.index)
    d = desired_exposure.reindex(all_publishers, fill_value=0).values
    a = actual_exposure.reindex(all_publishers, fill_value=0).values

    jsd = jensenshannon(d, a, base=2)
    jsd_divergence = jsd ** 2

    if jsd_divergence == 0:
        return np.inf
    return 1 / jsd_divergence


def fairness_penalty_rerank(hybrid_scores, item_ids, item_to_publisher, desired_exposure, beta, topn=20):
    """
    Score_fair(u,i) = Score_hyb(u,i) * V(p)

        u = a user
        i = a candidate item
        p = the publisher of item i
        Score_hyb(u,i) = the hybrid model's score for item i, for user u
        V(p) = a fairness multiplier for publisher p, computed as
            (desired_exposure(p) / actual_exposure(p)) ** beta
        beta = strength of the correction (0 = no adjustment, 1 = full correction)

    Simplified adaptation of Modani et al.'s fairness reranking. The original
    algorithm adjusts exposure live, one recommendation slot at a time, across
    all users at once. Here, we instead:
        1. Compute what publisher exposure would look like with no correction
           at all (using plain hybrid scores).
        2. Compare that actual exposure to each publisher's desired exposure
           (from publisher_capital, based on their contribution to the platform).
        3. Multiply every item's score by V(p): publishers under-served
           relative to their desired exposure get boosted, over-served
           publishers get dampened.
    """
    baseline_recs = [] #no correction, hybrid recommendations
    for u in range(hybrid_scores.shape[0]):
        row = hybrid_scores[u]
        k = min(topn, np.sum(row > -np.inf))
        if k == 0:
            baseline_recs.append([])
            continue
        top_cols = np.argpartition(-row, k - 1)[:k]
        baseline_recs.append([item_ids[c] for c in top_cols])

    exposure = actual_exposure(baseline_recs, item_to_publisher, k=topn)

    #make sure they have the same size
    all_publishers = desired_exposure.index.union(exposure.index) 
    desired_aligned = desired_exposure.reindex(all_publishers, fill_value=1e-6)
    actual_aligned = exposure.reindex(all_publishers, fill_value=1e-6)

    ratio = desired_aligned / actual_aligned
    ratio = ratio.clip(lower=0.1, upper=10)
    value_multiplier = (ratio ** beta).to_dict()

    print(f"Applying fairness reranking with beta={beta:.2f}: ")

    scores_fair = hybrid_scores.copy()
    for col, item in enumerate(item_ids):
        publisher = item_to_publisher.get(item)
        mult = value_multiplier.get(publisher, 1.0)
        scores_fair[:, col] = scores_fair[:, col] * mult

    return scores_fair


def check_beta(hybrid_scores_masked, item_ids, item_to_publisher, desired_exposure,
               user_ids, user_item_matrix, test_out_df, similarity_matrix, id_index,
               topn=20, betas=np.arange(0, 1.05, 0.1)):
    """Evaluate fairness reranking across a range of beta values"""

    results = []
    for beta in betas:
        scores_fair = fairness_penalty_rerank(hybrid_scores_masked, item_ids, item_to_publisher,
                                               desired_exposure, beta, topn)
        recs = recommend_hybrid(scores_fair, user_item_matrix, item_ids, topn)

        ndcg = ndcg_at_k(recs, user_ids, test_out_df, k=topn)
        recall = recall_at_k(recs, user_ids, test_out_df, k=topn)
        gini = item_gini(recs, k=topn)
        ils = intra_list_similarity(recs, similarity_matrix, id_index, k=topn)

        exposure = actual_exposure(recs, item_to_publisher, k=topn)
        fairness = publisher_fairness_score(desired_exposure, exposure)

        print(f"beta={beta:.2f} | ndcg={ndcg:.4f} | recall={recall:.4f} | "
              f"gini={gini:.4f} | ils={ils:.4f} | fairness={fairness:.4f}")
        results.append({"param": beta, "ndcg": ndcg, "recall": recall,
                         "gini": gini, "ils": ils, "fairness": fairness})

    return results