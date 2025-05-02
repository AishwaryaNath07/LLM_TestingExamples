import openai
from dotenv import load_dotenv
load_dotenv()
import os
# Initialize OpenAI and SentenceTransformer
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
from sentence_transformers import SentenceTransformer, util

# Initialize OpenAI client with new SDK format
# Define test cases
test_cases = [
    {
        "input": "What is the capital of France?",
        "expected_output": "Paris"
    },
    {
        "input": "78-2 is",
        "expected_output": "76"
    }
]

# Evaluation function (exact match for simplicity)
def is_correct(expected, actual):
    return expected.strip().lower() == actual.strip().lower()

# Function to call OpenAI
def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()



model = SentenceTransformer('all-MiniLM-L6-v2')

def is_semantically_similar(expected, actual, threshold=0.8):
    embeddings = model.encode([expected, actual], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity > threshold
# Run test suite
def run_tests():
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test['input']}")
        actual_output = call_openai(test["input"])
        correct = is_correct(test["expected_output"], actual_output)
        print(f"Expected: {test['expected_output']}")
        print(f"Actual:   {actual_output}")
        print(f"Result:   {'✅ PASS' if correct else '❌ FAIL'}\n")
        print("Semantic score:"+str(is_semantically_similar(test["expected_output"], actual_output, threshold=0.8)))

run_tests()

