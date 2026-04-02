from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def transform(image_size, augment=False):
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)



class CLIPLabeledDataset(Dataset):
    def __init__(self, root, image_size=256, augment=False):
        root = Path(root)

        self.classes = sorted(d.name for d in root.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            label = self.class_to_idx[cls]
            for img_path in (root / cls).iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))

        self.transform = transform(image_size, augment=augment)

    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class UnlabeledDataset(Dataset):
    def __init__(self, root, null_label, image_size=256, augment=False):
        root = Path(root)
        self.null_label = null_label
        self.samples = [
            p for p in root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        self.transform = transform(image_size, augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.samples[idx]).convert("RGB")), self.null_label


class CombinedDataset(Dataset):
    def __init__(self, labeled_root, unlabeled_roots, image_size=256, augment=False):
        self._labeled = CLIPLabeledDataset(labeled_root, image_size=image_size, augment=augment)
        null_label = self._labeled.num_classes   # one past the last real class

        self._unlabeled = []
        for root in unlabeled_roots:
            p = Path(root)
            if p.exists():
                ds = UnlabeledDataset(p, null_label=null_label, image_size=image_size, augment=augment)
                if len(ds):
                    self._unlabeled.append(ds)

        from torch.utils.data import ConcatDataset
        all_ds = [self._labeled] + self._unlabeled
        self._concat = ConcatDataset(all_ds)

    def num_classes(self):
        return self._labeled.num_classes

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        return self._concat[idx]


def get_dataloader(root, source="clip", image_size=256,
                   batch_size=32, num_workers=4, shuffle=True,
                   unlabeled_roots=None, augment=False):

    if source == "combined":
        dataset = CombinedDataset(root, unlabeled_roots or [], image_size=image_size, augment=augment)
    elif source == "clip":
        dataset = CLIPLabeledDataset(root, image_size=image_size, augment=augment)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset.num_classes
