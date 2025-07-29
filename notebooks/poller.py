import threading, time, queue, datetime, pynvml, csv, json, os

class GPUUtilSampler(threading.Thread):
    def __init__(self, run_meta: dict, out_dir: str, interval=0.1):
        super().__init__(daemon=True)
        self.interval, self.out_dir = interval, out_dir
        self.meta = run_meta
        self.rows = []           # in‑memory buffer

    def run(self):
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while True:
            util   = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem    = pynvml.nvmlDeviceGetUtilizationRates(h).memory
            ts     = datetime.datetime.now().timestamp()
            self.rows.append((ts, util, mem))
            time.sleep(self.interval)

    def flush(self):
        os.makedirs(self.out_dir, exist_ok=True)
        # write metrics
        with open(f"{self.out_dir}/util.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["ts", "gpu_util", "mem_util"]); wr.writerows(self.rows)
        # write metadata (hyper‑params, dataset path, git SHA, etc.)
        with open(f"{self.out_dir}/meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)