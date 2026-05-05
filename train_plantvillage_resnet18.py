import os
import copy
import time
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_species(label_name: str) -> str:
    return label_name.split("___")[0]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_split_dirs(dataset_root: str):
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    def pick(names):
        for name in names:
            p = root / name
            if p.is_dir():
                return str(p)
        return None

    train_dir = pick(["train"])
    val_dir = pick(["valid", "val", "validation"])
    test_dir = pick(["test"])

    if not train_dir or not val_dir or not test_dir:
        raise RuntimeError(
            f"Expected {root} to contain train/, valid/ (or val/), and test/ folders."
        )

    return train_dir, val_dir, test_dir


# -----------------------------
# Model helpers
# -----------------------------
def set_trainable_layers(model: nn.Module, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    if mode == "head":
        return
    elif mode == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif mode == "layer3_layer4":
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")


def build_model(num_classes: int, mode: str, device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    set_trainable_layers(model, mode)
    return model.to(device)


# -----------------------------
# Train / eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


def save_learning_curves(history: dict, out_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_f1"], label="Train")
    plt.plot(epochs, history["val_f1"], label="Val")
    plt.title("Macro F1")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------
def run_experiment(
    data_root: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    epochs: int,
    seed: int,
    run_modes: list[str],
) -> None:
    ensure_dir(output_dir)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_dir, val_dir, test_dir = find_split_dirs(data_root)

    # Transforms based on pretrained ResNet18 recipe
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Load train / valid / test directly from your folder structure
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)

    if val_dataset.class_to_idx != class_to_idx:
        raise RuntimeError("Validation classes do not match training classes.")
    if test_dataset.class_to_idx != class_to_idx:
        raise RuntimeError("Test classes do not match training classes.")

    print(f"Dataset root: {data_root}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    print(f"Test dir:  {test_dir}")
    print(f"Classes: {num_classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")
    print(f"Total images: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    results = []

    for mode in run_modes:
        print("\n" + "=" * 70)
        print(f"Training mode: {mode}")
        print("=" * 70)

        model = build_model(num_classes=num_classes, mode=mode, device=device)
        trainable_params = count_trainable_params(model)
        print(f"Trainable parameters: {trainable_params:,}")

        lr = 1e-3 if mode == "head" else 1e-4
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_f1 = -1.0
        best_epoch = -1

        history = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_f1": []
        }

        for epoch in range(epochs):
            start = time.time()

            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

            elapsed = time.time() - start
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        model.load_state_dict(best_model_wts)

        test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
            model, test_loader, criterion, device
        )

        pred_class_names = [class_names[i] for i in y_pred]
        true_class_names = [class_names[i] for i in y_true]
        pred_species = [extract_species(x) for x in pred_class_names]
        true_species = [extract_species(x) for x in true_class_names]
        species_acc = float(np.mean([p == t for p, t in zip(pred_species, true_species)]))

        mode_dir = Path(output_dir) / mode
        ensure_dir(mode_dir)

        ckpt_path = mode_dir / "best_model.pth"
        torch.save({
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "species_acc": species_acc,
        }, ckpt_path)

        curves_path = mode_dir / "learning_curves.png"
        save_learning_curves(history, str(curves_path))

        cm = confusion_matrix(y_true, y_pred)
        cm_path = mode_dir / "confusion_matrix.png"
        save_confusion_matrix(cm, str(cm_path), title=f"Confusion Matrix - {mode}")

        report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(mode_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        with open(mode_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        result_row = {
            "mode": mode,
            "trainable_params": trainable_params,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "species_acc": species_acc,
            "checkpoint": str(ckpt_path),
        }
        results.append(result_row)

        print("\nFinal results:")
        print(json.dumps(result_row, indent=2))

    results_df = pd.DataFrame(results)
    results_csv = Path(output_dir) / "results_summary.csv"
    results_df.to_csv(results_csv, index=False)

    print("\nSaved summary to:", results_csv)
    print(results_df)

def parse_args():
    parser = argparse.ArgumentParser(description="Train pretrained ResNet18 on PlantDiseaseDataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="PlantDiseaseDataset",
        help="Root folder containing train/, valid/, and test/"
    )
    parser.add_argument("--output_dir", type=str, default="outputs_plantvillage")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        type=str,
        default="head,layer4,layer3_layer4,all",
        help="Comma-separated fine-tuning modes"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    print("Using dataset root:", args.data_root)

    run_modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    run_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        seed=args.seed,
        run_modes=run_modes,
    )