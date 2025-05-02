import random
from datasets import load_dataset
from reclip.utils.model_recaption import recaption_with_model
from reclip.utils.icon_metadata import IconMetadata, IconShape, VisualStyle, ColorMode, ContainerType, DetailLevel

# Load dataset
dataset = load_dataset("andrewburns/hf_flat_icons", split="train")

# Generate 5 random indices
def five_random(start=0, end=1000):
    return random.sample(range(start, end + 1), 5)

nums = five_random(0, 1000)
print("Randomly selected indices:", nums)

# Extract examples using those indices
examples = [dataset[i] for i in nums]

# Display metadata and generate descriptions
for i, example in enumerate(examples):
    print(f"\nExample {i+1}:")
    print("Set:", example["set"])
    print("Title:", example["title"])
    print("Style:", example["style"])
    print("Tags:", example["tags"])

    print("Tags (input):", example["tags"])
    metadata = IconMetadata(tags=example["tags"].split(", "))
    print("Metadata:", metadata)
    prompt = metadata.to_prompt()
    print("Prompt:", prompt)

    description = recaption_with_model(prompt)
    print("Model Description:", description)
