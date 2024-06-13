import umap
import pke
import nltk
import numpy as np
import scipy.sparse as sp
import string
from semantic_similarity import get_specter2_embedding_keywords
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INTRODUCTION_VARS = ["introduction"]
CONCLUSION_VARS = ["conclusion", "conclusions"]

def text_preprocess(text):
    stoplist = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stoplist and word not in punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    preprocessed_text = ' '.join(words)

    return preprocessed_text

def get_view_texts(articles):
    bib_texts = []
    lex_texts = []
    key_texts = []
    for _, title, abstract, sections, bibs, _, _, _ in articles:
        bib_text = []
        lex_text = [title, abstract]
        key_text = [title, abstract]
        for text_segment, section in sections:
            section_lower = section.lower()
            lex_text.append(text_segment)
            if section_lower in CONCLUSION_VARS or section_lower in INTRODUCTION_VARS:
                key_text.append(text_segment)
        for bib in bibs:
            bib_text.extend([bib.get('title'), bib.get('venue')])
        bib_texts.append(' '.join(bib_text))    
        lex_texts.append(' '.join(lex_text))
        key_texts.append(' '.join(key_text))
    return bib_texts, lex_texts, key_texts

def scope_check(draft, corpus, model, tokenizer, device):
    extractor = pke.unsupervised.YAKE()
    articles = [draft]
    articles.extend(corpus)
    bib_texts, lex_texts, key_texts = get_view_texts(articles)
    bib_view = TFIDF_vector(bib_texts)
    lex_view = TFIDF_vector(lex_texts)
    articles_keywords = [keyword_extraction(extractor, key_text) for key_text in key_texts]
    key_view = []
    for article_keywords in articles_keywords:
        keywords = [item[0] for item in article_keywords]
        embeddings = get_specter2_embedding_keywords(keywords, model, tokenizer, device).tolist()
        key_view.append([item for sublist in embeddings for item in sublist])
    lex_norm, bib_norm, key_norm = normalize_data(lex_view, bib_view, key_view)
    is_within_radius = cluster_check(lex_norm, bib_norm, key_norm, cluster_n = 5)
    return is_within_radius

def cluster_check(lex, bib, key, cluster_n = 3):
    embeddings = np.hstack([lex, bib, key])
    draft_embedding = embeddings[0]
    texts_embedding = embeddings[1:]
    max_similarity = 0
    for embedding in texts_embedding:
        similarity = cosine_similarity(draft_embedding.reshape(1, -1), embedding.reshape(1, -1))
        max_similarity= max(max_similarity, similarity)
    print(max_similarity)
    kmeans = KMeans(n_clusters = cluster_n, random_state=42)
    labels = kmeans.fit_predict(texts_embedding)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    radii = []
    for i in range(cluster_n):
        cluster_points = texts_embedding[labels == i]
        centroid = centroids[i]
        distances = np.linalg.norm(cluster_points - centroid, axis = 1)
        radius = np.max(distances)
        radii.append(radius)
    predicted_cluster = kmeans.predict(draft_embedding.reshape(1, -1))[0]
    distance_to_centroid = np.linalg.norm(draft_embedding - centroids[predicted_cluster])
    is_within_radius = distance_to_centroid <= radii[predicted_cluster]
    print(labels)
    print(f"Predicted cluster: {predicted_cluster}")
    print(f"Distance to centroid: {distance_to_centroid}")
    print(f"Mean radius: {radii[predicted_cluster]}")
    print(f"Is within radius: {is_within_radius}")
    return is_within_radius

def reduce_dimensionality(embeddings, n_neighbors=5, n_components=3, min_dist=0.0, metric='euclidean'):
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors, 
        n_components=n_components, 
        min_dist=min_dist, 
        metric=metric)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    return reduced_embeddings

def keyword_extraction(extractor, text):
    stoplist = set(stopwords.words("english"))
    extractor.load_document(input = text,
                                language='en',
                                stoplist=stoplist,
                                normalization=None)
    extractor.candidate_selection(n=3)
    extractor.candidate_weighting(window=2, use_stems=True)
    keywords = extractor.get_n_best(n=30, threshold = 0.9)

    return keywords

def normalize_data(lex, bib, key):
    scaler_lex = StandardScaler()
    scaler_bib = StandardScaler()
    scaler_key = StandardScaler()
    lex_scal = scaler_lex.fit_transform(np.asarray(lex))
    bib_scal = scaler_bib.fit_transform(np.asarray(bib))
    key_scal = scaler_key.fit_transform(np.asarray(key))
    
    return lex_scal, bib_scal, key_scal

def TFIDF_vector(articles):
    vectorizer = TfidfVectorizer()
    preprocessed_texts = []
    for article in articles:
        preprocessed_texts.append(text_preprocess(article))
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts[1:])
    tfidf_draft = vectorizer.transform([preprocessed_texts[0]])
    tfidf_combined = sp.vstack((tfidf_draft, tfidf_matrix))
    tfidf_matrix_normalized = normalize(tfidf_combined, norm='l2', axis=1)
    dense_tfidf_matrix = tfidf_matrix_normalized.todense()

    return dense_tfidf_matrix