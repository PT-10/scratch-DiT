import argparse
import csv
import os
import sys
import pathlib

import numpy as np
from tqdm import tqdm

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations, calculate_frechet_distance
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_fid.fid_score import calculate_frechet_distance


def compute_stats(image_dir, dims=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # Collect all image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        str(p) for p in pathlib.Path(image_dir).rglob("*")
        if p.suffix.lower() in exts
    ]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    class SimpleImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = SimpleImageDataset(image_paths, transform)
    loader  = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    acts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  inception", leave=False):
            batch = batch.to(device)
            pred  = model(batch)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
            acts.append(pred.squeeze(3).squeeze(2).cpu().numpy())

    acts = np.concatenate(acts, axis=0)
    mu   = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def fid_from_stats(mu1, sigma1, mu2, sigma2):
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def parse_args():
    p = argparse.ArgumentParser(description="Compute FID scores for sweep configs")
    p.add_argument("--samples_dir", default="samples/",
                   help="root dir with generated samples: {samples_dir}/{config}/{step}/")
    p.add_argument("--real_dir",    default="data/data_labeled_v2/",
                   help="directory containing real training images")
    p.add_argument("--out",         default="fid_results.csv",
                   help="output CSV path")
    p.add_argument("--steps",       default=None,
                   help="comma-separated step checkpoints to include, e.g. '10000,30000'")
    p.add_argument("--real_stats_cache", default="real_stats.npz",
                   help="path to cache real-image inception statistics")
    p.add_argument("--dims",        type=int, default=2048,
                   help="InceptionV3 feature dimensionality (default 2048)")
    p.add_argument("--real_only",   action="store_true",
                   help="only compute and cache real image stats, then exit")
    p.add_argument("--configs",     default=None,
                   help="comma-separated config names to include, e.g. 'dit_s2,d6-h6-p2'")
    return p.parse_args()

def main():
    args = parse_args()

    # Parse requested steps
    wanted_steps = None
    if args.steps:
        wanted_steps = [int(s.strip()) for s in args.steps.split(",")]

    wanted_configs = None
    if args.configs:
        wanted_configs = {c.strip() for c in args.configs.split(",")}

    samples_dir = args.samples_dir

    # Discover configs and steps from samples directory
    if not os.path.isdir(samples_dir):
        print(f"samples_dir not found: {samples_dir}")
        sys.exit(1)

    # {config_name: {step: path}}
    config_map: dict[str, dict[int, str]] = {}

    for config_name in sorted(os.listdir(samples_dir)):
        config_path = os.path.join(samples_dir, config_name)
        if not os.path.isdir(config_path):
            continue
        if wanted_configs is not None and config_name not in wanted_configs:
            continue
        for step_str in sorted(os.listdir(config_path)):
            step_path = os.path.join(config_path, step_str)
            if not os.path.isdir(step_path):
                continue
            try:
                step = int(step_str)
            except ValueError:
                continue
            if wanted_steps is not None and step not in wanted_steps:
                continue
            config_map.setdefault(config_name, {})[step] = step_path

    if not config_map:
        print("No sample directories found matching the requested steps.")
        sys.exit(1)

    # Determine the set of all steps present
    all_steps = sorted({
        step
        for steps in config_map.values()
        for step in steps
    })
    if wanted_steps:
        # Preserve user-specified order
        all_steps = [s for s in wanted_steps if s in all_steps]

    print(f"Configs : {sorted(config_map.keys())}")
    print(f"Steps   : {all_steps}")

    # Real image statistics (cached) 
    cache_path = args.real_stats_cache
    if os.path.exists(cache_path):
        print(f"\nLoading cached real stats from {cache_path}")
        data      = np.load(cache_path)
        mu_real   = data["mu"]
        sigma_real = data["sigma"]
    else:
        print(f"\nComputing real image stats from {args.real_dir} …")
        if not os.path.isdir(args.real_dir):
            print(f"real_dir not found: {args.real_dir}")
            sys.exit(1)
        mu_real, sigma_real = compute_stats(args.real_dir, dims=args.dims)
        np.savez(cache_path, mu=mu_real, sigma=sigma_real)
        print(f"Cached → {cache_path}")

    if args.real_only:
        print("--real_only: done.")
        return

    # Compute FID for each config × step
    # results[config_name][step] = fid_value
    results: dict[str, dict[int, float]] = {}

    for config_name, step_paths in sorted(config_map.items()):
        results[config_name] = {}
        for step in all_steps:
            if step not in step_paths:
                results[config_name][step] = float("nan")
                continue

            img_dir = step_paths[step]
            n_imgs  = len([
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"\n[{config_name}] step {step:,}  ({n_imgs} images)")

            mu_gen, sigma_gen = compute_stats(img_dir, dims=args.dims)
            fid = fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen)
            results[config_name][step] = fid
            print(f"  FID = {fid:.2f}")

    # Build and print table
    step_headers = [f"FID@{s // 1000}k" for s in all_steps]

    # Sort by FID at final step ascending (NaN last)
    final_step = all_steps[-1]
    def sort_key(item):
        fid = item[1].get(final_step, float("nan"))
        return fid if not np.isnan(fid) else float("inf")

    sorted_results = sorted(results.items(), key=sort_key)

    # Column widths
    col_model = max(len("Model"), max(len(c) for c in results)) + 2
    col_fid   = max(len(h) for h in step_headers) + 2
    col_delta = len("ΔFID") + 2

    header_parts = [f"{'Model':<{col_model}}"]
    for h in step_headers:
        header_parts.append(f"{h:>{col_fid}}")
    if len(all_steps) >= 2:
        header_parts.append(f"{'ΔFID':>{col_delta}}")
    header = "".join(header_parts)
    sep    = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    csv_rows = []
    for config_name, step_fids in sorted_results:
        row_parts = [f"{config_name:<{col_model}}"]
        fid_vals  = []
        for step in all_steps:
            fid = step_fids.get(step, float("nan"))
            fid_vals.append(fid)
            if np.isnan(fid):
                row_parts.append(f"{'N/A':>{col_fid}}")
            else:
                row_parts.append(f"{fid:>{col_fid}.2f}")

        delta_str = ""
        if len(fid_vals) >= 2 and not np.isnan(fid_vals[0]) and not np.isnan(fid_vals[-1]):
            delta = fid_vals[-1] - fid_vals[0]
            delta_str = f"{delta:+.2f}"

        if len(all_steps) >= 2:
            row_parts.append(f"{delta_str:>{col_delta}}")

        print("".join(row_parts))

        csv_row = {"model": config_name}
        for step, fid in zip(all_steps, fid_vals):
            csv_row[f"fid_{step}"] = "" if np.isnan(fid) else f"{fid:.4f}"
        if len(all_steps) >= 2 and delta_str:
            csv_row["delta_fid"] = delta_str
        csv_rows.append(csv_row)

    print(sep)

    # Save CSV 
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
