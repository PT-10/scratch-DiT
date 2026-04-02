import os
import torch
from tqdm import tqdm

from model.autoencoder import AutoencoderKL
from dataloading.dataset import get_dataloader

VAE_SCALE   = 0.18215
_VAE_CONFIG = dict(
    ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
    in_channels=3, out_ch=3, z_channels=4, double_z=True, resolution=256,
)

DATA_ROOT    = "data/data_labeled_v2"
DATA_SOURCE  = "clip"
VAE_CKPT     = "pretrained-models/kl-f8-model.ckpt"
OUT_PATH     = "training-checkpoints/latents_dit_s2.pt"
BATCH_SIZE   = 32
NUM_WORKERS  = 4


def encode_dataset_to_cache(loader, num_classes, vae, device, cache_path):
    all_z, all_labels = [], []
    vae.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="encoding latents"):
            z = vae.encode(images.to(device)).sample() * VAE_SCALE
            all_z.append(z.cpu())
            all_labels.append(labels)
    all_z      = torch.cat(all_z)
    all_labels = torch.cat(all_labels)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save({"latents": all_z, "labels": all_labels, "num_classes": num_classes}, cache_path)
    print(f"saved {all_z.shape} → {cache_path}  ({all_z.numel()*4/1e6:.1f} MB)")
    return all_z, all_labels, num_classes


def load_latent_cache(cache_path):
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    return data["latents"], data["labels"], int(data["num_classes"])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKL(_VAE_CONFIG, embed_dim=4, ckpt_path=VAE_CKPT).to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    loader, num_classes = get_dataloader(
        DATA_ROOT, source=DATA_SOURCE, image_size=256,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        unlabeled_roots=[], augment=False,
    )

    encode_dataset_to_cache(loader, num_classes, vae, device, OUT_PATH)
