#Imports
import numpy as np
from fwls_hybrid import recommend_hybrid
from metrics import ndcg_at_k, recall_at_k, item_gini, intra_list_similarity

"""
Post-hoc reranking strategies applied to the hybrid scores:
- Popularity penalty (reduces popularity bias) 
- MMR (encourages diversity within each user's recommendation list).
"""

def compute_item_popularity(train_data, item_ids):
    """
    Count how many times each item appears in train_data,
    then normalize to [0, 1] so it's comparable across items regardless of
    dataset size
    """
    raw_counts = train_data["item_id"].value_counts()

    
    popularity = raw_counts.reindex(item_ids, fill_value=0).values.astype(float) #fill w/ 0 items that have zero training interactions

    max_pop = popularity.max()
    if max_pop > 0:
        popularity = popularity / max_pop  #normalize to [0, 1]

    return popularity


def popularity_penalty_rerank(hybrid_scores, item_ids, train_data, gamma, topn=20):
    """
    Score_pop(u,i) = Score_hyb(u,i) - gamma * Pop(i)
    Subtracts a popularity penalty from every item's score. 
    Higher gamma = stronger penalty against popular items.
    """
    popularity = compute_item_popularity(train_data, item_ids)

    print(f"Applying popularity penalty with gamma={gamma:.2f}")
    scores_pop = hybrid_scores - gamma * popularity  #broadcasts across all users

    return scores_pop

def check_gamma(hybrid_scores_masked, item_ids, train_data, user_ids, user_item_matrix,
                 test_out_df, similarity_matrix, id_index, topn=20,
                 gammas=np.arange(0, 1.05, 0.1)):
    """Evaluate popularity penalty across a range of gamma values"""
    results = []
    for gamma in gammas:
        scores_pop = popularity_penalty_rerank(hybrid_scores_masked, item_ids, train_data, gamma, topn)
        recs = recommend_hybrid(scores_pop, user_item_matrix, item_ids, topn)

        ndcg = ndcg_at_k(recs, user_ids, test_out_df, k=topn)
        recall = recall_at_k(recs, user_ids, test_out_df, k=topn)
        gini = item_gini(recs, k=topn)
        ils = intra_list_similarity(recs, similarity_matrix, id_index, k=topn)

        print(f"gamma={gamma:.2f} | ndcg={ndcg:.4f} | recall={recall:.4f} | gini={gini:.4f} | ils={ils:.4f}")
        results.append({"param": gamma, "ndcg": ndcg, "recall": recall, "gini": gini, "ils": ils})

    return results



def mmr_rerank_su(relevance_scores, similarity_matrix, candidate_idx, lam, topn=20):
    """
    Greedy MMR selection for one user.
    relevance_scores: full array of hybrid scores for this user
    similarity_matrix: item-item CB similarity matrix (LDA topic vectors) — used
        as a diversity metric, so recommendations are penalized for being
        content-similar to items already selected
    candidate_idx: indices of items eligible to be recommended
    lam: lambda, balance between relevance (high λ) and diversity (low λ)
    topn: number of items to select for this user
    """
    selected = []
    remaining = np.array(candidate_idx)

    for _ in range(min(topn, len(remaining))):
        if len(selected) == 0:
            #no items selected yet, so there's nothing to be
            #"similar to" — diversity term is 0
            diversity_penalty = np.zeros(len(remaining))
        else:
            
            diversity_penalty = similarity_matrix[np.ix_(remaining, selected)].max(axis=1)
            #how similar is candidate i to the most similar item already selected

        relevance = relevance_scores[remaining]
        mmr_scores = lam * relevance - (1 - lam) * diversity_penalty

        best_pos = np.argmax(mmr_scores)
        best_item = remaining[best_pos]

        selected.append(best_item)
        remaining = np.delete(remaining, best_pos)

    return selected


def mmr_rerank(hybrid_scores, similarity_matrix, item_ids, lam, topn=20):
    """
    Apply MMR reranking to every user
    """
    n_users = hybrid_scores.shape[0]
    print(f"Applying MMR reranking with lambda={lam:.2f} for {n_users} users")

    recommendations = []
    for u in range(n_users):
        user_scores = hybrid_scores[u]
        candidate_idx = np.where(user_scores > -np.inf)[0]

        if len(candidate_idx) == 0:
            recommendations.append([])
            continue

        selected_idx = mmr_rerank_su(user_scores, similarity_matrix, candidate_idx, lam, topn)
        recommendations.append([item_ids[i] for i in selected_idx])

    return recommendations

def check_lambda(hybrid_scores_masked, similarity_matrix, item_ids, user_ids,
                  test_out_df, id_index, topn=20, lambdas=np.arange(0, 1.05, 0.1)):
    """Evaluate MMR across a range of lambda values"""

    results = []
    for lam in lambdas:
        recs = mmr_rerank(hybrid_scores_masked, similarity_matrix, item_ids, lam, topn)

        ndcg = ndcg_at_k(recs, user_ids, test_out_df, k=topn)
        recall = recall_at_k(recs, user_ids, test_out_df, k=topn)
        gini = item_gini(recs, k=topn)
        ils = intra_list_similarity(recs, similarity_matrix, id_index, k=topn)

        print(f"lambda={lam:.2f} | ndcg={ndcg:.4f} | recall={recall:.4f} | gini={gini:.4f} | ils={ils:.4f}")
        results.append({"param": lam, "ndcg": ndcg, "recall": recall, "gini": gini, "ils": ils})

    return results