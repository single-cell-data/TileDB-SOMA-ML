import pandas as pd, matplotlib.pyplot as plt, glob, json, os
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--io_batch_size", type=int, default=4096,
                        help="Rows read from disk per IO batch")
    # you can expose any other hyper‑param the same way:
    return parser.parse_args()


def main():
    means, medians, labels = [], [], []
    args = get_args()

    for run_dir in glob.glob(f"runs/{args.io_batch_size}_*"):
        df = pd.read_csv(os.path.join(run_dir, "util.csv"))
        with open(os.path.join(run_dir, "meta.json")) as f:
            meta = json.load(f)

        df = df[df["gpu_util"] > 10]          # drop warm-up idle rows
        means.append(df["gpu_util"].mean())
        medians.append(df["gpu_util"].median())

        labels.append(f"sh={meta['shuffle_chunk_size']}")

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - w/2, means,   width=w, label="mean")
    plt.bar(x + w/2, medians, width=w, label="median")
    plt.ylabel("GPU util (%)")
    plt.xticks(x, labels)
    plt.title(f"GPU Utilization for Dataset in S3, IO Batch Size = {meta['io_batch_size']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{meta['io_batch_size']}_gpu_util_stats.png")


if __name__ == "__main__":
    main()