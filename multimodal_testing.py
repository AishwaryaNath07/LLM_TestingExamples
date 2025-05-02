import openai
import base64
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv()
import os
# Initialize OpenAI and SentenceTransformer
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small but effective

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Run multimodal test and get model response
def multimodal_test(prompt_text, image_path):
    base64_image = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Compare generated response with expected answer
def evaluate_response(generated, expected):
    embeddings = model.encode([generated, expected], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity_score, 4)

# ==== RUNNING THE TEST ====

image_path = "inputs/persian_pomerian.jpg"  # Replace with your image
prompt = "Describe the contents of this image clearly and completely."

expected_response = (
    "The image shows a fluffy black Persian cat and a small white Pomeranian dog sitting next to each other on a wooden floor. "
    "They appear calm and are looking in the different direction. The background has soft lighting."
)

# Run GPT-4 vision model
generated_response = multimodal_test(prompt, image_path)
print("\n🖼️ GPT-4 Response:\n", generated_response)

# Evaluate the output
similarity = evaluate_response(generated_response, expected_response)
print(f"\n✅ Similarity Score (0 to 1): {similarity}")
