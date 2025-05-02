
from reclip.pipeline.embed_and_score import compute_similarity

tests = [
    ("data/sample_images/apple.png", "A red apple with a round shape"),
    ("data/sample_images/star.png", "A yellow five-pointed star on a white background"),
    ("data/sample_images/cat.png", "A simple cartoon cat face with ears and eyes"),
]

for image_path, prompt in tests:
    try:
        similarity = compute_similarity(image_path, prompt)
        print(f"{image_path} vs '{prompt}': Similarity = {similarity:.4f}")
    except Exception as e:
        print(f"Error with {image_path}: {e}")
