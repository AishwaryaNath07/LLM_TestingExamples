from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, and decent accuracy

# Your sentences
sentences = [
    "The cat sat on the mat and stared at the window",
    "The cat stared at the window while sitting on the mat",
    "On the mat the cat sat and stared outside the window."
]

# Encode sentences to get embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine similarity matrix
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

# Print similarity matrix
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            print(f"Similarity between Sentence {i+1} and Sentence {j+1}: {cosine_scores[i][j]:.4f}")
