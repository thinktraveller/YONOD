import pandas as pd

# Load splits
test = pd.read_csv("test_1D_Aldehyde_1.csv")
train = pd.read_csv("train_1D_Aldehyde_1.csv")

# Extract unique (Aldehyde_1, bifunctional_reagent) pairs
test_pairs = set(tuple(x) for x in test[["Aldehyde_1", "bifunctional_reagent"]].drop_duplicates().values)
train_pairs = set(tuple(x) for x in train[["Aldehyde_1", "bifunctional_reagent"]].drop_duplicates().values)

# Compute overlap and test-only pairs
overlap = test_pairs.intersection(train_pairs)
test_only = test_pairs.difference(train_pairs)

# --- Summary ---
print("Number of unique pairs in TEST:", len(test_pairs))
print("Number of unique pairs in TRAIN:", len(train_pairs))
print("Number of overlapping pairs:", len(overlap))
print("Number of TEST-only pairs:", len(test_only))

# --- Print overlapping pairs ---
print("\n=== Overlapping pairs (in both TEST and TRAIN) ===")
for i, p in enumerate(sorted(overlap), start=1):
    print(f"{i}. {p}")

# --- Print TEST-only pairs ---
print("\n=== TEST-only pairs (held out correctly) ===")
for i, p in enumerate(sorted(test_only), start=1):
    print(f"{i}. {p}")

