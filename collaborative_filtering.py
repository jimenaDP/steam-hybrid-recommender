#Imports
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

"""Collaborative filtering recommender using ItemKNN: item-item similarity
based on shared user interaction patterns"""

def build_binary_matrix(train_data, test_in_df):
    """Sort and join all users and items appearing in train and fold-in test data."""
    all_users = sorted(set(train_data["user_id"]) | set(test_in_df["user_id"]))
    all_items = sorted(set(train_data["item_id"]) | set(test_in_df["item_id"]))

    #sanity check: sets can't contain duplicates, but this catches upstream bugs
    #if all_users/all_items are ever built a different way later
    assert len(all_users) == len(set(all_users)), "duplicate users found in all_users"
    assert len(all_items) == len(set(all_items)), "duplicate items found in all_items"

    user2row = {u: i for i, u in enumerate(all_users)}
    item2col = {it: i for i, it in enumerate(all_items)}
    return all_users, all_items, user2row, item2col

def build_user_item_matrix_cf(interactions, user2row, item2col):
    """Binary user x item matrix"""
    rows, cols = [], []

    for _, row in interactions.iterrows():
        if row.user_id in user2row and row.item_id in item2col:
            rows.append(user2row[row.user_id])
            cols.append(item2col[row.item_id])

    data = np.ones(len(rows))
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user2row), len(item2col)))

    print(f"User-item matrix for CF: {matrix.shape[0]} users x {matrix.shape[1]} items, "
          f"{matrix.nnz} interactions")
    return matrix

def item_cosine_similarity(user_item_matrix):
    """Cosine similarity between items, based on shared user interactions"""
    item_vectors = user_item_matrix.T
    similarity_matrix = cosine_similarity(item_vectors)

    #sanity check: a similarity matrix must be square and symmetric
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "similarity matrix is not square"
    assert np.allclose(similarity_matrix, similarity_matrix.T), "similarity matrix is not symmetric"

    return similarity_matrix

def topk_filter(similarity_matrix, k=200):
    """Keep only the top-k most similar items per item (ItemKNN), zero out the rest"""
    n_items = similarity_matrix.shape[0]
    filtered = np.zeros_like(similarity_matrix)

    print(f"Calculating the {k} nearest neighbours for {n_items} items...")

    for i in range(n_items):
        row = similarity_matrix[i]
        top_k_idx = np.argpartition(-row, min(k, n_items - 1))[:k]
        filtered[i, top_k_idx] = row[top_k_idx]

    print(f"Top-{k} item-item similarity matrix calculated")
    return filtered

def recommend_cf(user_item_matrix, similarity_matrix, all_items, topn=20):
    """Score each candidate by similarity to items the user already interacted with.
    Already-seen items are excluded. Returns a list aligned with the matrix's row order"""
    n_users = user_item_matrix.shape[0]
    print(f"Generating top-{topn} CF recommendations for {n_users} users...")

    scores = np.asarray(user_item_matrix @ similarity_matrix)

    seen_mask = user_item_matrix.toarray().astype(bool)
    scores[seen_mask] = -np.inf

    recommendations = []
    for u in range(n_users):
        user_scores = scores[u]
        k = min(topn, np.sum(user_scores > -np.inf))
        if k == 0:
            recommendations.append([])
            continue
        top_cols = np.argpartition(-user_scores, k - 1)[:k]
        top_cols = top_cols[np.argsort(-user_scores[top_cols])]
        recommendations.append([all_items[c] for c in top_cols])

    n_empty = sum(1 for r in recommendations if len(r) == 0)
    print(f"Done. {n_empty} users received no recommendations.")

    return recommendations

##################ONLY USED FOR THE HYBRID MODEL################################
def score_cf(user_item_matrix, similarity_matrix):
    """Raw CF scores (no masking, no top-N) - needed for the hybrid model"""
    return np.asarray(user_item_matrix @ similarity_matrix)
################################################################################

def build_cf_model(train_data, test_in_df, k=200, topn=20):
    """Run the full CF pipeline"""
    all_users, all_items, user2row, item2col = build_binary_matrix(train_data, test_in_df)

    X_train = build_user_item_matrix_cf(train_data, user2row, item2col)
    X_test_in = build_user_item_matrix_cf(test_in_df, user2row, item2col)

    similarity_matrix = item_cosine_similarity(X_train)
    similarity_matrix = topk_filter(similarity_matrix, k=k)

    recommendations = recommend_cf(X_test_in, similarity_matrix, all_items, topn=topn)

    return recommendations, all_users, all_items, similarity_matrix