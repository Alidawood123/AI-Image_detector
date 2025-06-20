import sys
from transformers import pipeline
from PIL import Image

# Usage: python image_detector.py path_to_image.jpg
if len(sys.argv) != 2:
    print("Usage: python image_detector.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Load image
image = Image.open(image_path).convert("RGB")

# Load the Organika/sdxl-detector model from Hugging Face
classifier = pipeline(
    "image-classification",
    model="Organika/sdxl-detector"
)

# Run the classifier
raw_results = classifier(image)
results = list(raw_results) if raw_results else []

# Pretty print results
print("\n================ AI Image Detection Results ================\n")
if results:
    print(f"{'Label':<25} | {'Score':>8}")
    print("-" * 38)
    for result in results:
        if isinstance(result, dict) and 'label' in result and 'score' in result:
            label = result.get('label')
            score = result.get('score')
            if label is not None and score is not None:
                print(f"{label:<25} | {score*100:7.2f}%")
            else:
                print(f"Label or score missing in result: {result}")
        else:
            print(f"Unexpected result format: {result}")
    # Highlight the top result
    top = results[0]
    if isinstance(top, dict) and 'label' in top and 'score' in top:
        label = top.get('label')
        score = top.get('score')
        print("\nPrediction:")
        if label is not None and score is not None:
            print(f"  ==> This image is most likely: \033[1m{label}\033[0m (confidence: {score*100:.2f}%)\n")
        else:
            print("  ==> Top result missing label or score.")
    else:
        print("\nPrediction: Unexpected result format for top result.")
else:
    print("No results returned. Please check your image or model.")
print("===========================================================\n")