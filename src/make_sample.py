from __future__ import annotations
import pandas as pd
from pathlib import Path

N_ROWS = 300

src = Path("data/processed/credit_default.csv")
dst = Path("tests/data/sample_credit.csv")
dst.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
df.head(N_ROWS).to_csv(dst, index=False)

print(f"Sample saved to {dst} | shape={df.head(N_ROWS).shape}")
