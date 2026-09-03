#Imports
import re
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import corpus2dense
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import nltk
from nltk.corpus import stopwords
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

#stopwords in english
STOPWORDS = set(stopwords.words("english"))


def concatenate_text(extended_games_cleaned):
    """Concatenate item_name + text columns into one 'text' field per game"""
    text_cols = ["item_name", "short_description", "about_the_game",
                 "detailed_description", "reviews"]
    extended_games_cleaned["text"] = extended_games_cleaned[text_cols].agg(" ".join, axis=1)
    return extended_games_cleaned

def preprocess_for_lda(text):
    """Preprocesses a text string for LDA (Latent Dirichlet Allocation).
        The function returns a list of clean tokens by:
          - converting text to lowercase
          - removing HTML tags
          - removing non-alphabetic characters
          - removing stopwords
          - and filtering out short words
        """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return tokens

def tokenize(extended_games_cleaned):
    extended_games_cleaned["tokens"] = extended_games_cleaned["text"].apply(preprocess_for_lda)

    #Sanity check: check games with empty token lists
    n_empty = (extended_games_cleaned["tokens"].apply(len) == 0).sum()
    print(f"{n_empty} games have zero tokens AFTER PREPROCESSING")
    assert n_empty < len(extended_games_cleaned) * 0.05, "too many empty token lists — check text pipeline"

    return extended_games_cleaned

def train_lda(tokens, num_topics=20, seed=42):
    """Train LDA on tokenized game descriptions."""
    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  #filter rare words that aren't meaningful

    corpus = [dictionary.doc2bow(t) for t in tokens] #we create a bag of words for every game

    print(f"Training LDA model with {num_topics} topics on {len(corpus)} documents...")

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary,
                          random_state=seed)

    print("Training done!")

    #sanity check: no topic should be completely unused across the corpus
    topic_totals = np.zeros(num_topics)
    for doc in corpus:
        for topic_id, prob in lda_model.get_document_topics(doc, minimum_probability=0):
            topic_totals[topic_id] += prob
    n_dead_topics = (topic_totals == 0).sum()
    print(f"{n_dead_topics} topic(s) have zero probability mass across all documents")
    assert n_dead_topics == 0, f"{n_dead_topics} dead topic(s) found — consider lowering num_topics"

    return lda_model, dictionary, corpus

def similarity(lda_model, corpus, num_topics):
    """Get each game's topic distribution vector, then apply cosine similarity"""
    topic_vectors = corpus2dense(
        lda_model[corpus], num_terms=num_topics
    ).T  #(games, topics)

    similarity_matrix = cosine_similarity(topic_vectors)
    return similarity_matrix, topic_vectors

def map_ids(item_ids):
    """Maps each game's item_id with its row position in the similarity matrix."""
    id_index = {item_id: i for i, item_id in enumerate(item_ids)}
    index_id = {i: item_id for item_id, i in id_index.items()}
    return id_index, index_id


def build_user_item_matrix_cb(hist, user_ids, id_index):
    """User x CB-item matrix from fold-in data"""
    rows, cols = [], []
    user2row = {u: i for i, u in enumerate(user_ids)}

    for _, row in hist.iterrows():
        if row.user_id in user2row and row.item_id in id_index:
            rows.append(user2row[row.user_id])
            cols.append(id_index[row.item_id])

    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(id_index)))


def recommend_cb(user_item_matrix, similarity_matrix, index_id, topn=20):
    """Score each candidate by its similarity to items the user has already interacted with.
    Already-seen items are excluded from recommendations."""
    n_users = user_item_matrix.shape[0]
    print(f"Generating top-{topn} CB recommendations for {n_users} users...")
    scores = np.asarray(user_item_matrix @ similarity_matrix)

    seen_mask = user_item_matrix.toarray().astype(bool) #exclude the games the user already played
    scores[seen_mask] = -np.inf

    recommendations = []
    for u in range(scores.shape[0]):
        user_scores = scores[u]
        k = min(topn, np.sum(user_scores > -np.inf))
        if k == 0:
            recommendations.append([])
            continue
        top_rows = np.argpartition(-user_scores, k - 1)[:k]
        top_rows = top_rows[np.argsort(-user_scores[top_rows])]
        recommendations.append([index_id[r] for r in top_rows])
    n_empty = sum(1 for r in recommendations if len(r) == 0)
    print(f"Done. {n_empty} users received no recommendations")

    return recommendations

##################ONLY USED FOR THE HYBRID MODEL################################
def score_cb(user_item_matrix, similarity_matrix):
    """Raw CB scores (no masking, no top-N) — needed for th hybrid model"""
    return np.asarray(user_item_matrix @ similarity_matrix)
################################################################################

def build_cb_model(reduced_eg, num_topics=20, seed=42):
    """Run the full CB pipeline"""
    reduced_eg = concatenate_text(reduced_eg)
    reduced_eg = tokenize(reduced_eg)

    lda_model, dictionary, corpus = train_lda(reduced_eg["tokens"], num_topics, seed)
    similarity_matrix, topic_vectors = similarity(lda_model, corpus, num_topics)

    id_index, index_id = map_ids(reduced_eg["item_id"].tolist())

    #sanity check: similarity matrix must align with the item index
    assert similarity_matrix.shape[0] == len(id_index), \
        f"similarity matrix has {similarity_matrix.shape[0]} rows but id_index has {len(id_index)} items"
    print(f"CB model built: {similarity_matrix.shape[0]} items, {num_topics} topics")

    return reduced_eg, lda_model, similarity_matrix, id_index, index_id



