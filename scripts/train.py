"""Train a LumaNet variant, optionally distilling from a larger teacher.

Examples:
    python scripts/train.py --name teacher --width 32 --iters 8000
    python scripts/train.py --name student --width 12 --teacher checkpoints/teacher.pt
    python scripts/train.py --name learned --arch learned --width 32 --iters 8000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumaforge.data import CropDataset
from lumaforge.isp.pipeline import ISPConfig
from lumaforge.metrics import isp_loss, psnr
from lumaforge.models.lumanet import LumaNet, NetConfig, count_parameters


def load_model(path: Path) -> tuple[LumaNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = LumaNet(NetConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


@torch.no_grad()
def validate(model: LumaNet, loader: DataLoader) -> float:
    model.eval()
    scores = []
    for packed, noise, base, target in loader:
        output = model(packed, noise, base).clamp(0, 1)
        for prediction, truth in zip(output.numpy(), target.numpy()):
            scores.append(psnr(prediction, truth))
    model.train()
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/crops"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--arch", choices=["hybrid", "learned"], default="hybrid")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--no-noise-map", action="store_true")
    parser.add_argument("--no-color-head", action="store_true")
    parser.add_argument("--no-degrade", action="store_true")
    parser.add_argument("--no-color-loss", action="store_true")
    parser.add_argument("--no-classical-denoise", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    config = NetConfig(
        width=args.width,
        hybrid=args.arch == "hybrid",
        use_noise_map=not args.no_noise_map,
        use_color_head=not args.no_color_head,
    )
    isp = ISPConfig(denoise=not args.no_classical_denoise)
    model = LumaNet(config).train()
    print(f"{args.name}: {count_parameters(model):,} params {config}", flush=True)

    teacher = None
    if args.teacher:
        teacher, _ = load_model(args.teacher)
        teacher.requires_grad_(False)

    train_set = CropDataset(
        args.data, "train", args.crop, degrade=not args.no_degrade, isp=isp,
        epoch_size=args.iters * args.batch,
    )
    val_set = CropDataset(args.data, "val", 192, augment=False, degrade=False, isp=isp)
    val_set.index = val_set.index[::6]
    val_set.epoch_size = len(val_set.index)
    loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(val_set, 8, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def schedule(step: int) -> float:
        warmup = 200
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(args.iters - warmup, 1)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    started = time.time()
    color_weight = 0.0 if args.no_color_loss else 0.02
    running = 0.0

    for step, (packed, noise, base, target) in enumerate(loader, 1):
        output = model(packed, noise, base)
        loss = isp_loss(output, target, color_weight=color_weight)
        if teacher is not None:
            with torch.no_grad():
                guide = teacher(packed, noise, base)
            loss = (1 - args.distill_weight) * loss + args.distill_weight * F.l1_loss(output, guide)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += float(loss)

        if step % args.val_every == 0 or step == args.iters:
            score = validate(model, val_loader)
            elapsed = time.time() - started
            mean_loss = running / (step % args.val_every or args.val_every)
            running = 0.0
            history.append({"step": step, "val_psnr": score, "loss": mean_loss, "seconds": elapsed})
            print(f"{args.name} step {step}/{args.iters} loss {mean_loss:.4f} "
                  f"val {score:.2f} dB {elapsed / step:.2f}s/it", flush=True)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": config.to_dict(),
                            "isp": isp.__dict__, "val_psnr": best, "step": step},
                           args.out / f"{args.name}.pt")
            (args.out / f"{args.name}.json").write_text(json.dumps(
                {"args": {k: str(v) for k, v in vars(args).items()}, "history": history,
                 "params": count_parameters(model)}, indent=1))
    print(f"TRAIN_DONE {args.name} best {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
