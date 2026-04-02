import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import clip
import umap
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CACHE     = "embeddings.npy"
DEFAULT_PATHS     = "embeddings_paths.json"
DEFAULT_UMAP      = "embeddings_umap2d.npy"
DEFAULT_ASSIGN    = "cluster_plots/cluster_assignments.json"
DEFAULT_PLOT_DIR  = "cluster_plots"


def find_images(root):
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    return paths

def cmd_embed(args):
    cache = Path(args.cache)
    paths_file = Path(args.paths)

    if cache.exists() and paths_file.exists() and not args.force:
        print(f"Cache already exists at {cache}. Use --force to re-embed.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    all_paths = find_images(args.image_dir)
    print(f"Found {len(all_paths):,} images in {args.image_dir}")

    batch_size = 256
    all_feats  = []
    valid_paths = []

    for i in tqdm(range(0, len(all_paths), batch_size), desc="embedding", unit="batch"):
        batch_paths = all_paths[i : i + batch_size]
        imgs = []
        batch_valid = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                imgs.append(img)
                batch_valid.append(str(p))
            except Exception:
                continue
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().float().numpy())
        valid_paths.extend(batch_valid)

    feats = np.concatenate(all_feats, axis=0)
    np.save(str(cache), feats)
    with open(paths_file, "w") as f:
        json.dump(valid_paths, f)

    print(f"Embeddings : {feats.shape}  → {cache}")
    print(f"Paths      : {len(valid_paths):,}  → {paths_file}")


def cmd_cluster(args):
    feats = np.load(args.cache)
    with open(args.paths) as f:
        paths = json.load(f)
    print(f"Loaded {feats.shape[0]:,} embeddings")

    feats = normalize(feats)

    # PCA to 50 dims before clustering - improves Euclidean distance quality
    print("Fitting PCA (512 → 50)...")
    pca = PCA(n_components=50, random_state=42)
    feats = pca.fit_transform(feats)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # UMAP 2D — cache it so re-runs with different k are fast
    umap_cache = Path(args.umap_cache)
    if umap_cache.exists() and not args.force:
        print(f"Loading cached UMAP from {umap_cache}")
        emb2d = np.load(str(umap_cache))
    else:
        print("Running UMAP (this takes a few minutes for 90k images)...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            random_state=42,
            verbose=True,
            low_memory=True,
        )
        emb2d = reducer.fit_transform(feats)
        np.save(str(umap_cache), emb2d)
        print(f"UMAP saved → {umap_cache}")

    # Elbow curve
    print("Computing elbow curve (k=4..20)...")
    k_range = range(4, 21)
    inertias = []
    for k in tqdm(k_range, desc="elbow", unit="k"):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=4096)
        km.fit(feats)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, "o-", linewidth=2)
    ax.set_xlabel("k  (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-means Elbow Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "elbow.png", dpi=150)
    plt.close(fig)
    print(f"Elbow plot → {out_dir / 'elbow.png'}")

    # Final clustering
    k = args.k
    print(f"Clustering with k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500)
    labels = km.fit_predict(feats)

    # Save assignments
    assignments = {p: int(l) for p, l in zip(paths, labels)}
    assign_path = out_dir / "cluster_assignments.json"
    with open(assign_path, "w") as f:
        json.dump(assignments, f)
    print(f"Assignments → {assign_path}")

    # UMAP scatter
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap="tab20", s=0.8, alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Cluster", shrink=0.8)
    ax.set_title(f"UMAP — {k} clusters  |  {len(paths):,} images", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_clusters.png", dpi=150)
    plt.close(fig)
    print(f"UMAP scatter → {out_dir / 'umap_clusters.png'}")

    # Per-cluster image grids
    n_samples = 20
    cols, rows = 5, 4
    print(f"Generating cluster grids ({k} clusters × {n_samples} samples)...")
    for cluster_id in tqdm(range(k), desc="cluster grids", unit="cluster"):
        cluster_paths = [p for p, l in zip(paths, labels) if l == cluster_id]
        sample_paths  = cluster_paths[:n_samples]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        fig.suptitle(
            f"Cluster {cluster_id}  ({len(cluster_paths):,} images)", fontsize=13
        )
        for ax in axes.flat:
            ax.axis("off")
        for i, p in enumerate(sample_paths):
            try:
                img = Image.open(p).convert("RGB").resize((128, 128))
                axes.flat[i].imshow(img)
            except Exception:
                pass
        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{cluster_id:02d}.png", dpi=100)
        plt.close(fig)

    # Summary
    print(f"\nCluster summary:")
    for cid in range(k):
        n = int((labels == cid).sum())
        print(f"  cluster {cid:2d} : {n:6,} images")
    print(f"\nReview plots in {out_dir}/")
    print(f'Then create mapping.json: {{"0": "beach", "3": "mountain", ...}}')
    print(f"Unmapped clusters are discarded during export.")


def cmd_export(args):
    with open(args.assignments) as f:
        assignments = json.load(f)
    with open(args.mapping) as f:
        mapping = json.load(f)  # {"0": "beach", "3": "mountain", ...}

    out_dir = Path(args.out_dir)
    counts  = {}
    skipped = 0

    for path, cluster_id in tqdm(assignments.items(), desc="exporting", unit="img"):
        key = str(cluster_id)
        if key not in mapping:
            skipped += 1
            continue
        class_name = mapping[key]
        dst_dir = out_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(path).name
        if not dst.exists():
            shutil.copy2(path, dst)
        counts[class_name] = counts.get(class_name, 0) + 1

    print(f"\nExported to {out_dir}:")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}: {n:,} images")
    print(f"Skipped (unmapped): {skipped:,}")


def main():
    p = argparse.ArgumentParser(description="CLIP clustering for image datasets")
    sub = p.add_subparsers(dest="cmd", required=True)

    # embed
    e = sub.add_parser("embed", help="CLIP-encode images and cache embeddings")
    e.add_argument("--image_dir",  required=True, help="root dir of images (searched recursively)")
    e.add_argument("--cache",      default=DEFAULT_CACHE,  help="output .npy file for embeddings")
    e.add_argument("--paths",      default=DEFAULT_PATHS,  help="output .json file for image paths")
    e.add_argument("--force",      action="store_true",    help="overwrite existing cache")

    # cluster
    c = sub.add_parser("cluster", help="UMAP + k-means, generate plots")
    c.add_argument("--cache",       default=DEFAULT_CACHE,   help="embeddings .npy")
    c.add_argument("--paths",       default=DEFAULT_PATHS,   help="paths .json")
    c.add_argument("--umap_cache",  default=DEFAULT_UMAP,    help="cached UMAP 2D coords .npy")
    c.add_argument("--k",           type=int, required=True,  help="number of clusters")
    c.add_argument("--out_dir",     default=DEFAULT_PLOT_DIR, help="output dir for plots")
    c.add_argument("--force",       action="store_true",      help="recompute UMAP even if cached")

    # export
    x = sub.add_parser("export", help="copy images into class folders")
    x.add_argument("--assignments", default=DEFAULT_ASSIGN,  help="cluster_assignments.json")
    x.add_argument("--mapping",     required=True,           help='JSON: {"0": "beach", ...}')
    x.add_argument("--out_dir",     required=True,           help="output data_labeled dir")

    args = p.parse_args()
    {"embed": cmd_embed, "cluster": cmd_cluster, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()