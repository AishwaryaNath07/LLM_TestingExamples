

import openai
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI client with new SDK format
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY "))

# Test prompt
test_prompt = "Summarize the causes of World War I."

# Function to call the LLM
def generate_response(prompt, temperature=0.0):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature  # Lower temperature for deterministic output
    )
    return response.choices[0].message.content.strip()

# Run repeatability test
def run_repeatability_test(prompt, runs=5, temperature=0.0):
    results = []
    for i in range(runs):
        output = generate_response(prompt, temperature)
        results.append(output)
        print(f"\n--- Response {i+1} ---\n{output}")
    return results

# Execute test
responses = run_repeatability_test(test_prompt, runs=3, temperature=0.0)

# Check for unique responses
unique_responses = list(set(responses))
print(f"\nTotal unique responses: {len(unique_responses)} out of {len(responses)}")
