import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from config import Config
from utils import set_seed, SimpleLogger, save_ckpt
from dataset import MultiModalPTDataset, make_patient_folds, split_by_fold
from model import MMDiffuzzy
from eval import evaluate

def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "pid":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--fold", type=int, default=-1)
    args = ap.parse_args()

    cfg = Config().ensure()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
        os.makedirs(cfg.out_dir, exist_ok=True)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.fold_index = args.fold

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    all_paths = MultiModalPTDataset.discover(cfg.data_dir)
    folds = make_patient_folds(all_paths, cfg.folds, cfg.seed)

    fold_indices = list(range(cfg.folds)) if cfg.fold_index < 0 else [cfg.fold_index]

    for fi in fold_indices:
        run_dir = os.path.join(cfg.out_dir, f"fold_{fi}")
        os.makedirs(run_dir, exist_ok=True)
        logger = SimpleLogger(run_dir)

        train_paths, test_paths = split_by_fold(all_paths, folds[fi])

        rng = torch.Generator().manual_seed(cfg.seed + fi)
        perm = torch.randperm(len(train_paths), generator=rng).tolist()
        train_paths = [train_paths[i] for i in perm]
        n_val = max(1, int(0.15 * len(train_paths)))
        val_paths = train_paths[:n_val]
        tr_paths = train_paths[n_val:]

        train_ds = MultiModalPTDataset(cfg.data_dir, tr_paths)
        val_ds = MultiModalPTDataset(cfg.data_dir, val_paths)
        test_ds = MultiModalPTDataset(cfg.data_dir, test_paths)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn
        )

        model = MMDiffuzzy(cfg).to(device)
        optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best = -1e9
        best_path = os.path.join(run_dir, "best.pt")

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            steps = 0

            for batch in train_loader:
                for k in ["wsi_latent", "wsi_lr", "genomics", "y", "time", "censor"]:
                    batch[k] = batch[k].to(device)

                optim.zero_grad(set_to_none=True)
                total, parts, _, _ = model.loss(batch)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

                running += float(total.item())
                steps += 1

            train_loss = running / max(1, steps)

            val_metrics = evaluate(model, val_loader, device)
            score = val_metrics["f1"] + val_metrics["cindex"]

            logger.log({
                "fold": fi,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "val_cindex": val_metrics["cindex"],
                "score": score,
            })

            if score > best:
                best = score
                save_ckpt(best_path, model, optim, epoch, best, cfg.__dict__)

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

        test_metrics = evaluate(model, test_loader, device)
        logger.log({
            "fold": fi,
            "best_score": best,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_f1": test_metrics["f1"],
            "test_cindex": test_metrics["cindex"],
        })
        print(fi, test_metrics)

if __name__ == "__main__":
    main()