
from reclip.data.hf_flat_icons_loader import load_hf_flat_icons, get_image_and_label
from reclip.pipeline.openclip_pipeline import compute_similarity_openclip

dataset = load_hf_flat_icons(limit=5)
print(dataset[0].keys())

print("🔍 Testing OpenCLIP on hf_flat_icons dataset:")
for i, example in enumerate(dataset):
    try:
        image, label = get_image_and_label(example)
        file_path = f"data/sample_images/icon_{i}_{label}.png"
        image.save(file_path)
        prompt = f"An icon about {label}"
        score = compute_similarity_openclip(file_path, prompt)
        print(f"[{label}] → '{prompt}': Similarity = {score:.4f}")
    except Exception as e:
        print(f"❌ Error for item {i} ({example['label']}): {e}")
