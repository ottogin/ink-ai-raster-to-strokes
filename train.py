"""Training script for the ink stroke predictor.

Usage:
    uv run python train.py --data path/to/dataset.npz
    uv run python train.py --data path/to/dataset.npz --epochs 100 --lr 1e-4
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from model import InkStrokePredictor
from dataset import create_dataloaders
from losses import mdn_loss
from metrics import compute_all_metrics


def make_stroke_input(trajectory):
    """Shift target right by 1 to create decoder input (teacher forcing).

    Input at t=0 is zeros, input at t=i is target at t=i-1.
    """
    B, T, C = trajectory.shape
    zeros = torch.zeros(B, 1, C, device=trajectory.device)
    return torch.cat([zeros, trajectory[:, :-1, :]], dim=1)


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    total_loc = 0
    total_pen = 0
    n_batches = len(loader)

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        trajectory = batch["trajectory"].to(device)
        lengths = batch["length"].to(device)

        stroke_in = make_stroke_input(trajectory)
        mdn_params = model(image, stroke_in)

        loss_dict = mdn_loss(mdn_params, trajectory, lengths)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_loc += loss_dict["loc_loss"].item()
        total_pen += loss_dict["pen_loss"].item()

        print(
            f"\r  Epoch {epoch}/{total_epochs} [{step+1}/{n_batches}] "
            f"loss={loss.item():.4f} (loc={loss_dict['loc_loss'].item():.4f}, pen={loss_dict['pen_loss'].item():.4f})",
            end="", flush=True,
        )

    print()  # newline after progress
    return {
        "loss": total_loss / n_batches,
        "loc_loss": total_loc / n_batches,
        "pen_loss": total_pen / n_batches,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_loc = 0
    total_pen = 0
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        trajectory = batch["trajectory"].to(device)
        lengths = batch["length"].to(device)

        stroke_in = make_stroke_input(trajectory)
        mdn_params = model(image, stroke_in)
        loss_dict = mdn_loss(mdn_params, trajectory, lengths)

        total_loss += loss_dict["loss"].item()
        total_loc += loss_dict["loc_loss"].item()
        total_pen += loss_dict["pen_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "loc_loss": total_loc / n_batches,
        "pen_loss": total_pen / n_batches,
    }


@torch.no_grad()
def evaluate_samples(model, loader, device, num_samples=50):
    """Run autoregressive sampling and compute metrics."""
    model.eval()
    all_metrics = []

    for batch in loader:
        image = batch["image"].to(device)
        trajectory = batch["trajectory"]
        lengths = batch["length"]

        pred_strokes, pred_lengths = model.sample(image, max_len=trajectory.shape[1])
        pred_strokes = pred_strokes.cpu().numpy()

        for i in range(len(image)):
            if len(all_metrics) >= num_samples:
                break
            pred = pred_strokes[i, :pred_lengths[i].item()]
            target = trajectory[i, :lengths[i].item()].numpy()
            if len(pred) < 2 or len(target) < 2:
                continue
            m = compute_all_metrics(pred, target)
            all_metrics.append(m)

        if len(all_metrics) >= num_samples:
            break

    if not all_metrics:
        return {}

    avg = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg[key] = float(np.mean(values))
        avg[f"{key}_std"] = float(np.std(values))
    return avg


def print_eval_metrics(metrics, prefix=""):
    if not metrics:
        print(f"{prefix}No valid samples for evaluation")
        return
    print(
        f"{prefix}"
        f"DTW={metrics.get('dtw', 0):.4f} (±{metrics.get('dtw_std', 0):.4f}), "
        f"Chamfer={metrics.get('chamfer', 0):.4f} (±{metrics.get('chamfer_std', 0):.4f}), "
        f"SSIM={metrics.get('ssim', 0):.4f} (±{metrics.get('ssim_std', 0):.4f}), "
        f"PenAcc={metrics.get('pen_accuracy', 0):.3f} (±{metrics.get('pen_accuracy_std', 0):.3f})"
    )


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loc_loss"], label="train")
    axes[1].plot(epochs, history["val_loc_loss"], label="val")
    axes[1].set_title("Location Loss (NLL)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["train_pen_loss"], label="train")
    axes[2].plot(epochs, history["val_pen_loss"], label="val")
    axes[2].set_title("Pen Loss (BCE)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train ink stroke predictor")
    parser.add_argument("--data", type=str, default="/Users/artemlukoyanov/Documents/Proga/Ilumni/playground/SundaiClub/weeks/inkai/ink-ai-hack-playground/notebooks/data/generated/dataset.npz", help="Path to dataset.npz")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_mixtures", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=5, help="Run sampling eval every N epochs")
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of samples for eval metrics")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size (for debugging)")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(
        f"Train: {len(train_loader.dataset)} samples, "
        f"Val: {len(val_loader.dataset)} samples, "
        f"Test: {len(test_loader.dataset)} samples"
    )

    # Model
    model = InkStrokePredictor(
        feature_dim=args.feature_dim,
        num_mixtures=args.num_mixtures,
        num_layers=args.num_layers,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    history = {
        "train_loss": [], "train_loc_loss": [], "train_pen_loss": [],
        "val_loss": [], "val_loc_loss": [], "val_pen_loss": [],
        "eval_metrics": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_metrics = validate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        # Record history
        for key in ["loss", "loc_loss", "pen_loss"]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"train: {train_metrics['loss']:.4f} (loc={train_metrics['loc_loss']:.4f}, pen={train_metrics['pen_loss']:.4f}) | "
            f"val: {val_metrics['loss']:.4f} (loc={val_metrics['loc_loss']:.4f}, pen={val_metrics['pen_loss']:.4f}) | "
            f"lr={lr:.2e}"
        )

        # Periodic sampling eval on val set
        if epoch % args.eval_every == 0:
            eval_metrics = evaluate_samples(model, val_loader, device, num_samples=args.eval_samples)
            history["eval_metrics"].append({"epoch": epoch, "split": "val", **eval_metrics})
            print_eval_metrics(eval_metrics, prefix="  Val eval: ")

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, output_dir / "best.pt")

        # Save latest
        save_checkpoint(model, optimizer, epoch, val_metrics, output_dir / "latest.pt")

    # --- Final evaluation on test set using best checkpoint ---
    print("\n" + "=" * 60)
    print("Final evaluation on TEST set (using best checkpoint)")
    print("=" * 60)

    best_ckpt = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']}")

    # Test loss
    test_loss = validate(model, test_loader, device)
    print(
        f"Test loss: {test_loss['loss']:.4f} "
        f"(loc={test_loss['loc_loss']:.4f}, pen={test_loss['pen_loss']:.4f})"
    )

    # Test sampling metrics
    test_eval = evaluate_samples(model, test_loader, device, num_samples=args.eval_samples)
    print_eval_metrics(test_eval, prefix="Test metrics: ")

    # Save final results
    final_results = {
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_eval,
    }
    history["final_test"] = final_results

    # Save training curves and history
    plot_training_curves(history, output_dir / "training_curves.png")
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} (epoch {best_ckpt['epoch']})")
    print(f"Checkpoints and results saved to {output_dir}")


if __name__ == "__main__":
    main()
