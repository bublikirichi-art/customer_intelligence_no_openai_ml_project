import sys
import pandas as pd

submission_path = sys.argv[1]
test_path = sys.argv[2] if len(sys.argv) > 2 else "data/test.csv"

allowed = {
    "fit_and_sizing",
    "material_and_quality",
    "style_and_appearance",
    "overall_wearability_and_value",
}

sub = pd.read_csv(submission_path)
test = pd.read_csv(test_path)

assert list(sub.columns) == ["id", "recommended", "reviewer_segment"], sub.columns.tolist()
assert len(sub) == len(test), (len(sub), len(test))
assert sub["id"].astype(str).tolist() == test["id"].astype(str).tolist()
assert set(sub["recommended"].unique()).issubset({0, 1})
assert set(sub["reviewer_segment"].unique()).issubset(allowed)
print("OK", submission_path)
