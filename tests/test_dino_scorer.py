
from reclip.pipeline.dino_scorer import ReclipScorer
from reclip.models.reclip_dino_model import get_random_icon
from reclip.utils.model_recaption import recaption_with_model

def test_dino_scorer():
    print("🧪 Running DINOv2 scoring test...")
    scorer = ReclipScorer()
    image, tags = get_random_icon()
    prompt = recaption_with_model(", ".join(tags))
    print("Prompt:", prompt)
    score = scorer.score_similarity(image, prompt)
    print(f"✅ Cosine similarity score: {score:.4f}")
    assert -1.0 <= score <= 1.0, "Similarity must be a valid cosine value between -1 and 1"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_dino_scorer()
