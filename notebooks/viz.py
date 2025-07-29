import pandas as pd, matplotlib.pyplot as plt, glob, json, os
import numpy as np

def trim_leading_zeros(df: pd.DataFrame, col: str = "gpu_util", thresh: int = 0):
    """
    Remove rows until the first value of *col* exceeds *thresh*.
    Returns a copy (original left untouched).
    """
    idx = np.argmax(df[col].to_numpy() > thresh)  # index of first non‑zero
    return df.iloc[idx:].reset_index(drop=True)


plt.figure(figsize=(10, 5))

for run_dir in glob.glob("runs/run_*"):
    df = pd.read_csv(os.path.join(run_dir, "util.csv"))
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)

    df = trim_leading_zeros(df, col="gpu_util", thresh=0)   # <<─ NEW

    # normalise time axis
    df["rel_t"] = df["ts"] - df["ts"].iloc[0]

    label = (f"bs={meta['batch_size']} | "
             f"store={'s3' if 's3://' in meta['dataset_uri'] else 'local'}")
    plt.plot(df["rel_t"], df["gpu_util"], label=label, alpha=0.8)

plt.xlabel("Seconds since first non‑zero util")
plt.ylabel("GPU util (%)")
plt.legend()
plt.tight_layout()
plt.savefig("gpu_util_runs.png")
plt.show()
