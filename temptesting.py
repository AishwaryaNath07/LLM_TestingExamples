import os
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = openai.OpenAI(api_key=api_key)

# Function to test different temperature values
def test_temperature(prompt, temperatures=[0.0, 0.5, 1.0, 1.5]):
    responses = {}
    for temp in temperatures:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        responses[temp] = response.choices[0].message.content  # Updated response parsing
    
    return responses

# Define a sample prompt
prompt = "Write me a joke on astronauts"

# Run the test and print results
results = test_temperature(prompt)
for temp, response in results.items():
    print(f"\nTemperature {temp}:")
    print(response)
