import openai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv()
import os
# Initialize OpenAI and SentenceTransformer
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example test cases
test_cases = [
    {"input": "Who is the president of the United States?", "expected": "Joe Biden"},
    {"input": "What is the capital of Germany?", "expected": "Berlin"},
    {"input": "5 multiplied by 3?", "expected": "15"},
]

# Call OpenAI API
def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Compare using semantic similarity
def is_semantically_correct(expected, actual, threshold=0.8):
    embeddings = model.encode([expected, actual], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity >= threshold, similarity

# Run accuracy test
def run_accuracy_test():
    total = len(test_cases)
    passed = 0

    for idx, test in enumerate(test_cases, 1):
        print(f"\nTest Case {idx}")
        prompt = test["input"]
        expected = test["expected"]

        actual = call_openai(prompt)
        correct, score = is_semantically_correct(expected, actual)

        print(f"Prompt   : {prompt}")
        print(f"Expected : {expected}")
        print(f"Actual   : {actual}")
        print(f"Similarity: {score:.2f}")
        print("Result   :", "✅ PASS" if correct else "❌ FAIL")

        if correct:
            passed += 1

    accuracy = passed / total * 100
    print(f"\n🔍 Accuracy: {passed}/{total} = {accuracy:.2f}%")

run_accuracy_test()
