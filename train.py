from pathlib import Path
import numpy as np
import pandas as pd

import modal
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm


from model import AudioCNN


app = modal.App("audio-cnn-train")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
    .run_commands([
        "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
    ])
    .add_local_python_source("model")
)

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

TARGET_SR = 22050 

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

        # FIX #1: Proper train/val split
        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        self.classes = sorted(self.metadata["category"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, row["label"]



def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume, "/models": model_volume},
    timeout=60 * 60 * 3,
)
def train():
    esc50_dir = Path("/opt/esc50-data")

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025,
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(30),
        T.TimeMasking(80),
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025,
        ),
        T.AmplitudeToDB(),
    )

    train_ds = ESC50Dataset(
        esc50_dir, esc50_dir / "meta" / "esc50.csv",
        split="train", transform=train_transform
    )

    val_ds = ESC50Dataset(
        esc50_dir, esc50_dir / "meta" / "esc50.csv",
        split="test", transform=val_transform
    )

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioCNN(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-3,
        epochs=100,
        steps_per_epoch=len(train_dl),
    )

    best_acc = 0.0

    print("🚀 Training started")

    for epoch in range(100):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/100"):
            x, y = x.to(device), y.to(device)

            if np.random.rand() < 0.7:
                x, y_a, y_b, lam = mixup_data(x, y)
                out = model(x)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                out = model(x)
                loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Accuracy = {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_ds.classes,
                    "accuracy": acc,
                },
                "/models/best_model.pth",
            )
            print("✅ Best model saved")

    print(f"🎯 Training complete. Best accuracy: {best_acc:.2f}%")


@app.local_entrypoint()
def main():
    train.remote()
