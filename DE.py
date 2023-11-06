import gensim
from gensim.models import KeyedVectors
import requests
import gzip
import shutil

# Step 1: Download the pretrained Word2Vec vectors from the provided link
download_url = "https://drive.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
destination_path = "GoogleNews-vectors-negative300.bin.gz"

response = requests.get(download_url, stream=True)
with open(destination_path, 'wb') as file:
    shutil.copyfileobj(response.raw, file)

# Step 2: Load the Word2Vec embeddings and store them as a flat file
# Make sure to change the 'limit' parameter according to your requirement
limit = 1000000  # Number of vectors to load

wv = KeyedVectors.load_word2vec_format(destination_path, binary=True, limit=limit)
output_path = 'vectors.csv'
wv.save_word2vec_format(output_path)

# Step 3: Continue working with the flat file ('vectors.csv')

# Clean up the downloaded compressed file if needed
shutil.rmtree(destination_path)



import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# Load the pre-trained Word2Vec embeddings (use the path to your CSV file)
wv = gensim.models.KeyedVectors.load_word2vec_format('vectors.csv', binary=False)

# Load phrases from a CSV file (e.g., 'phrases.csv')
phrases_df = pd.read_csv('phrases.csv')

# Step a) Assign each word in each phrase a Word2Vec embedding
def get_phrase_vector(phrase):
    word_vectors = [wv[word] for word in phrase.split() if word in wv]
    if not word_vectors:
        return None
    # Approximate the phrase vector as the normalized sum of word vectors
    phrase_vector = np.sum(word_vectors, axis=0)
    return phrase_vector / np.linalg.norm(phrase_vector)

# Step b) Calculate L2 distance (Euclidean distance) or Cosine distance
def calculate_distances(phrase, all_phrases, distance_metric='cosine'):
    target_vector = get_phrase_vector(phrase)
    if target_vector is None:
        return []

    all_vectors = [get_phrase_vector(p) for p in all_phrases]
    valid_indices = [i for i, vector in enumerate(all_vectors) if vector is not None]

    valid_vectors = [all_vectors[i] for i in valid_indices]
    valid_phrases = [all_phrases[i] for i in valid_indices]

    if distance_metric == 'cosine':
        distances = cosine_distances([target_vector], valid_vectors).flatten()
    elif distance_metric == 'euclidean':
        distances = euclidean_distances([target_vector], valid_vectors).flatten()

    return list(zip(valid_phrases, distances))

# Step c) On-the-fly execution
def find_closest_match(user_input, all_phrases, distance_metric='cosine'):
    distances = calculate_distances(user_input, all_phrases, distance_metric)
    if not distances:
        return None

    closest_match, min_distance = min(distances, key=lambda x: x[1])
    return closest_match, min_distance

# Example usage:
user_input = "Your user-input phrase"
closest_phrase, distance = find_closest_match(user_input, phrases_df['phrase'])

print(f"Closest phrase: {closest_phrase}")
print(f"Distance: {distance}")