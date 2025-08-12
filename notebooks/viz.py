# import pandas as pd, matplotlib.pyplot as plt, glob, json, os
# import numpy as np

# def trim_leading_zeros(df: pd.DataFrame, col: str = "gpu_util", thresh: int = 0):
#     """
#     Remove rows until the first value of *col* exceeds *thresh*.
#     Returns a copy (original left untouched).
#     """
#     idx = np.argmax(df[col].to_numpy() > thresh)  # index of first non‑zero
#     return df.iloc[idx:].reset_index(drop=True)


# plt.figure(figsize=(10, 5))

# for run_dir in glob.glob("runs/4096_*"):
#     df = pd.read_csv(os.path.join(run_dir, "util.csv"))
#     with open(os.path.join(run_dir, "meta.json")) as f:
#         meta = json.load(f)

#     df = trim_leading_zeros(df, col="gpu_util", thresh=10)   # <<─ NEW

#     # normalise time axis
#     df["rel_t"] = df["ts"] - df["ts"].iloc[0]

#     label = (f"io_bs={meta['io_batch_size']} | "
#              f"sc_sz={meta['shuffle_chunk_size']} | "
#              f"store={'s3' if 's3://' in meta['dataset_uri'] else 'local'}")
#     plt.plot(df["rel_t"], df["gpu_util"], label=label, alpha=0.8)

# plt.xlabel("Seconds since first non‑zero util")
# plt.ylabel("GPU util (%)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("gpu_util_runs.png")
# plt.show()

import pandas as pd, matplotlib.pyplot as plt, glob, json, os, numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--io_batch_size", type=int, default=4096,
                        help="Rows read from disk per IO batch")

    parser.add_argument("--shuffle_chunk_size", type=int, default=4096,
                        help="Shuffle Chunk Size")
    
    parser.add_argument("--across", type=bool, default=False)
                        
    # you can expose any other hyper‑param the same way:
    return parser.parse_args()

def main():
    plt.figure(figsize=(11, 5))

    args = get_args()

    print(args.across)

    if args.across:
        print(f"runs/*_{args.shuffle_chunk_size}_*")
        for run_dir in glob.glob(f"runs/*_{args.shuffle_chunk_size}_*"):
            print(run_dir)
            df = pd.read_csv(os.path.join(run_dir, "util.csv"))
            with open(os.path.join(run_dir, "meta.json")) as f:
                meta = json.load(f)

            # drop idle pre-warm-up rows
            df = df[df["gpu_util"] > 10].reset_index(drop=True)

            df["rel_t"] = df["ts"] - df["ts"].iloc[0]           # seconds since first util
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="s")    # convert to datetime
            df.set_index("ts_dt", inplace=True)

            # smooth = df["gpu_util"].resample("0.01s").mean()       # 1-second means
            # smooth = smooth.rolling(window=10, min_periods=1, center=True).mean()  # 5-s MA

            smooth = (
                df["gpu_util"]
                .rolling("10s", min_periods=1)   # <- 10-second *time* window
                .mean()
            )


            label = f"io={meta['io_batch_size']} | sh={meta['shuffle_chunk_size']}"
            plt.plot(smooth.index - smooth.index[0], smooth.values, label=label, alpha=0.8)  # subtract start for x-axis
    else:

        for run_dir in glob.glob(f"runs/s3_runs/*_{args.io_batch_size}_*"):
            df = pd.read_csv(os.path.join(run_dir, "util.csv"))
            with open(os.path.join(run_dir, "meta.json")) as f:
                meta = json.load(f)

            # drop idle pre-warm-up rows
            df = df[df["gpu_util"] > 10].reset_index(drop=True)

            df["rel_t"] = df["ts"] - df["ts"].iloc[0]           # seconds since first util
            df["ts_dt"] = pd.to_datetime(df["ts"], unit="s")    # convert to datetime
            df.set_index("ts_dt", inplace=True)

            # smooth = df["gpu_util"].resample("0.01s").mean()       # 1-second means
            # smooth = smooth.rolling(window=10, min_periods=1, center=True).mean()  # 5-s MA

            smooth = (
                df["gpu_util"]
                .rolling("10s", min_periods=1)   # <- 10-second *time* window
                .mean()
            )


            label = f"io={meta['io_batch_size']} | sh={meta['shuffle_chunk_size']}"
            plt.plot(smooth.index - smooth.index[0], smooth.values, label=label, alpha=0.8)  # subtract start for x-axis


    plt.xlabel("Seconds since first util >10 %")
    plt.ylabel(f"GPU Util (%) — (1s mean, 5s rolling)")
    if args.across:
        plt.title(f"GPU Util Time Series - SH = {args.shuffle_chunk_size}, IN MEM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"sh_{args.shuffle_chunk_size}_gpu_util_smooth.png")
        plt.savefig(os.path.join("images", f"sh_{args.shuffle_chunk_size}_gpu_util_smooth.png"))

    else:
        plt.title(f"GPU Util Time Series - IO CHUNK SIZE = {args.io_batch_size}, IN MEM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("images", f"dataset_io_{args.io_batch_size}_gpu_util_smooth.png"))
    



if __name__ == "__main__":
    main()
