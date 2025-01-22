# Import required libraries
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
from itertools import combinations

# --- NER z użyciem spaCy ---
# Załaduj polski model spaCy
nlp = spacy.load("pl_core_news_sm")

# Przykładowy tekst w języku polskim
text = "Mary miała małą owieczkę, a potem przekazała talerz Johnowi w Warszawie. Patryk natomiast miał ryby."

# Przetwarzanie tekstu
doc = nlp(text)

# Wydobywanie jednostek nazwanych (NER)
entities = [ent.text for ent in doc.ents]
print("Rozpoznane jednostki nazwane:", entities)

# --- Ekstrakcja słów kluczowych z użyciem TF-IDF ---
texts = [text]  # Możesz dodać więcej tekstów, jeżeli chcesz analizować zbiór tekstów

# Załadowanie stop words z pliku
with open('stopwords_pl.txt', 'r', encoding='utf-8') as file:
    stop_words_polish = [line.strip() for line in file]

# Ekstrakcja cech TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words_polish)
X = vectorizer.fit_transform(texts)

# Pobierz słowa kluczowe z TF-IDF
features = vectorizer.get_feature_names_out()
print("Słowa kluczowe:", features)

# --- Łączenie jednostek nazwanych (NER) i słów kluczowych ---
# Stwórz listę pojęć do analizy
concepts = set(list(features))
concepts.update(entities)
print("Pojęcia do analizy (po dodaniu nazw własnych):", concepts)

# --- Uzyskanie embeddingów za pomocą modelu LaBSE ---
# Załaduj model i tokenizer LaBSE
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

# Funkcja do uzyskania embeddingów dla pojęć
def get_concept_embeddings(concepts):
    embeddings = []
    for concept in concepts:
        inputs = tokenizer(concept, return_tensors="pt")
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1)
            embeddings.append(embedding)
    return embeddings

# Uzyskaj embeddingi dla pojęć
embeddings = get_concept_embeddings(concepts)
concepts_list = list(concepts)

# Tworzenie krawędzi na podstawie podobieństw embeddingów
edges = []
for (i, j) in combinations(range(len(concepts_list)), 2):
    sim = cosine_similarity(embeddings[i], embeddings[j]).item()
    if sim > 0.55:  # ustawienie progu podobieństwa
        edges.append({
            "node_1": concepts_list[i],
            "node_2": concepts_list[j],
            "relation": "semantic similarity",
            "weight": sim
        })

# --- Tworzenie krawędzi na podstawie zależności składniowych ---
for token in doc:
    if token.text in concepts:
        for child in token.children:
            if child.text in concepts:
                edges.append({
                    "node_1": token.text,
                    "node_2": child.text,
                    "relation": child.dep_,
                    "weight": 1
                })

# --- Agregowanie krawędzi ---
df_edges = pd.DataFrame(edges)
print("Krawędzie przed agregowaniem:", df_edges)

# Agregowanie krawędzi przez łączenie relacji i sumowanie wag
df_aggregated = (
    df_edges.groupby(['node_1', 'node_2'])
    .agg({"relation": lambda x: ", ".join(set(x)), "weight": "sum"})
    .reset_index()
)

# --- Tworzenie grafu w NetworkX ---
G = nx.Graph()
for _, row in df_aggregated.iterrows():
    G.add_node(row["node_1"])
    G.add_node(row["node_2"])
    G.add_edge(row["node_1"], row["node_2"], title=row["relation"], weight=row["weight"])

# --- Wizualizacja grafu za pomocą PyVis ---
net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)
net.show("graph_of_concepts.html")

print("Graph saved as 'graph_of_concepts.html'. Open it in a browser to view the interactive visualization.")
