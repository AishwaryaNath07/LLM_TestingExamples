import torch
import clip
from PIL import Image
import os

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_image_from_file(filename):
    """Load image from current working directory"""
    # Verify file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Image file '{filename}' not found in current directory")
    
    # Verify it's an image file
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    if not filename.lower().endswith(valid_extensions):
        raise ValueError(f"File '{filename}' is not a supported image format")
    
    return Image.open(filename)

def get_clip_image_embeddings(image):
    """Get CLIP embeddings for an image"""
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features

def calculate_clip_similarity(image1, image2):
    """
    Calculate CLIP similarity score between two images
    Returns a value between 0 (dissimilar) and 1 (very similar)
    """
    # Get embeddings for both images
    embeddings1 = get_clip_image_embeddings(image1)
    embeddings2 = get_clip_image_embeddings(image2)
    
    # Calculate cosine similarity
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    similarity = cos_sim(embeddings1, embeddings2).item()
    
    # Convert to 0-1 scale
    return (similarity + 1) / 2

# Example usage
if __name__ == "__main__":
    # Example filenames - replace with your actual image filenames
    image_file1 = "inputs/cat1.jpg"
    image_file2 = "inputs/cat2.jpg"
    
    try:
        img1 = load_image_from_file(image_file1)
        img2 = load_image_from_file(image_file2)
        
        similarity_score = calculate_clip_similarity(img1, img2)
        print(f"CLIP similarity score between {image_file1} and {image_file2}: {similarity_score:.4f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
