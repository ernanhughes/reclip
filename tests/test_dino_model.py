
from reclip.models.reclip_dino_model import ReclipVisionModel, get_random_icon

def test_dino_embedding():
    print("🧪 Running DINOv2-Small test...")
    model = ReclipVisionModel()
    image, tags = get_random_icon()
    embedding = model.embed_image(image)
    print("Tags:", tags)
    print("Embedding shape:", embedding.shape)
    assert embedding.shape[-1] == 384, "Expected 384-dimensional embedding from DINOv2-Small"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_dino_embedding()
