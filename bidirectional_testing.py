import openai
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from difflib import SequenceMatcher
from dotenv import load_dotenv
load_dotenv()
import os

'''BLEU Score: Measures n-gram precision; higher = closer match.

ROUGE-1: Measures word overlap (recall).

ROUGE-L: Measures longest common subsequence.

SequenceMatcher: Measures similarity as a quick heuristic.'''
# Helper to call GPT-3.5
nltk.download("punkt_tab")

# Set your OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def gpt_style_transfer(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for style rewriting."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
# Similarity score using difflib
def sequence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# BLEU score
def bleu_score(reference, hypothesis):
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    return sentence_bleu([reference_tokens], hypothesis_tokens)

# ROUGE score
def rouge_scores(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)

# --- Style Transfer Test ---

original_text = "I do not think we can proceed with the plan because we cannot secure approval."

# Step 1: Formal → Informal
prompt1 = f"Rewrite the following text in an informal, conversational style:\n\n{original_text}"
informal_text = gpt_style_transfer(prompt1)

# Step 2: Informal → Formal
prompt2 = f"Rewrite the following text in a formal and professional tone:\n\n{informal_text}"
reconstructed_text = gpt_style_transfer(prompt2)

# Evaluation
similarity = sequence_similarity(original_text, reconstructed_text)
bleu = bleu_score(original_text, reconstructed_text)
rouge = rouge_scores(original_text, reconstructed_text)

# --- Output ---
print("Original (Formal):\n", original_text)
print("\nInformal Version:\n", informal_text)
print("\nReconstructed (Formal):\n", reconstructed_text)

print("\n--- Evaluation Scores ---")
print(f"Sequence Similarity: {similarity:.2f}")
print(f"BLEU Score: {bleu:.2f}")
print(f"ROUGE-1 Score: {rouge['rouge1'].fmeasure:.2f}")
print(f"ROUGE-L Score: {rouge['rougeL'].fmeasure:.2f}")
