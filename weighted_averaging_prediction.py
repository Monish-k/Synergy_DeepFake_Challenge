import json
from glob import glob
from collections import defaultdict

# Path pattern to 6 prediction files
file_pattern = "indivisual_predictions_json/*.json"

# Load all JSON files
files = sorted(glob(file_pattern))  # ensure consistent order
print(f"Found {len(files)} prediction files.")
print(files)

# Define weights (corresponding to file order)
weights = [0.3, 0.2, 0.11, 0.11, 0.13, 0.13]
print("Weights:", weights)

# Load predictions
all_predictions = []
for f in files:
    with open(f, "r") as infile:
        data = json.load(infile)
        all_predictions.append({d["index"]: d["prediction"] for d in data})

# Collect all indices
indices = sorted(all_predictions[0].keys())

# Perform weighted voting
final_predictions = []
for idx in indices:
    scores = defaultdict(float)
    for i, preds in enumerate(all_predictions):
        pred = preds[idx]
        scores[pred] += weights[i]
    # Choose prediction with highest weighted score
    final_prediction = max(scores, key=scores.get)
    final_predictions.append({"index": idx, "prediction": final_prediction})

# Save output
with open("final_prediction_weighted.json", "w") as outfile:
    json.dump(final_predictions, outfile, indent=4)

print("Weighted majority voting complete! Saved as 'prasanna.kotyal_prediction.json'")