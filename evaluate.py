"""Standalone evaluation script.

Usage:
    uv run python evaluate.py --data path/to/dataset.npz --checkpoint checkpoints/best.pt
    uv run python evaluate.py --data path/to/dataset.npz --checkpoint checkpoints/best.pt --num_samples 50 --visualize
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import InkStrokePredictor
from dataset import create_dataloaders
from metrics import compute_all_metrics, render_trajectory


@torch.no_grad()
def run_evaluation(model, loader, device, num_samples=50):
    """Evaluate model on samples and return per-sample metrics + predictions."""
    model.eval()
    results = []

    for batch in loader:
        image = batch["image"].to(device)
        trajectory = batch["trajectory"]
        lengths = batch["length"]
        words = batch["word"]

        pred_strokes, pred_lengths = model.sample(image, max_len=trajectory.shape[1])
        pred_np = pred_strokes.cpu().numpy()

        for i in range(len(image)):
            if len(results) >= num_samples:
                break
            pred = pred_np[i, :pred_lengths[i].item()]
            target = trajectory[i, :lengths[i].item()].numpy()
            if len(pred) < 2 or len(target) < 2:
                continue
            m = compute_all_metrics(pred, target)
            results.append({
                "word": words[i],
                "metrics": m,
                "pred_deltas": pred,
                "target_deltas": target,
                "image": image[i].cpu().numpy(),
            })

        if len(results) >= num_samples:
            break

    return results


def visualize_predictions(results, save_path, num_show=16):
    """Grid of input image + predicted vs target trajectory."""
    num_show = min(num_show, len(results))
    fig, axes = plt.subplots(num_show, 3, figsize=(12, 3 * num_show))
    if num_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_show):
        r = results[i]

        # Input image
        img = np.transpose(r["image"], (1, 2, 0))  # CHW → HWC
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input: '{r['word']}'", fontsize=9)
        axes[i, 0].axis("off")

        # Target trajectory render
        target_img = render_trajectory(r["target_deltas"])
        axes[i, 1].imshow(target_img, cmap="gray")
        axes[i, 1].set_title("Target trajectory", fontsize=9)
        axes[i, 1].axis("off")

        # Predicted trajectory render
        pred_img = render_trajectory(r["pred_deltas"])
        axes[i, 2].imshow(pred_img, cmap="gray")
        m = r["metrics"]
        axes[i, 2].set_title(
            f"Predicted (DTW={m['dtw']:.3f}, SSIM={m['ssim']:.3f})",
            fontsize=9,
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ink stroke predictor")
    parser.add_argument("--data", type=str, default="/Users/artemlukoyanov/Documents/Proga/Ilumni/playground/SundaiClub/weeks/inkai/ink-ai-hack-playground/notebooks/data/generated/dataset.npz", help="Path to dataset.npz")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_mixtures", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--visualize", action="store_true", help="Save visualization grid")
    parser.add_argument("--num_visualize", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = InkStrokePredictor(
        feature_dim=args.feature_dim,
        num_mixtures=args.num_mixtures,
        num_layers=args.num_layers,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load data — evaluate on the test split
    _, _, test_loader = create_dataloaders(
        args.data, batch_size=args.batch_size, val_split=0.1, test_split=0.1,
    )
    print(f"Evaluating on {len(test_loader.dataset)} test samples")

    # Run eval
    results = run_evaluation(model, test_loader, device, num_samples=args.num_samples)

    # Aggregate metrics
    metric_keys = list(results[0]["metrics"].keys())
    aggregated = {}
    for key in metric_keys:
        values = [r["metrics"][key] for r in results]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    print(f"\n{'Metric':<15} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-" * 45)
    for key, stats in aggregated.items():
        print(f"{key:<15} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['median']:>8.4f}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")

    # Visualization
    if args.visualize:
        visualize_predictions(results, output_dir / "predictions.png", num_show=args.num_visualize)


if __name__ == "__main__":
    main()
