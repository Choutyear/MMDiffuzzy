import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MultiModalPTDataset
from metrics import accuracy_f1, c_index

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    all_risk, all_time, all_event = [], [], []
    losses = []

    for batch in loader:
        # allow either "event" or "censor"
        keys = ["wsi_latent", "wsi_lr", "genomics", "y", "time"]
        for k in keys:
            batch[k] = batch[k].to(device)

        if "event" in batch:
            batch["event"] = batch["event"].to(device)
            event = batch["event"]
        else:
            batch["censor"] = batch["censor"].to(device)
            # default: censor=1 means censored -> event = 1 - censor
            event = 1.0 - batch["censor"]

        total, parts, logits, risk = model.loss(batch)
        losses.append(float(total.item()))

        all_logits.append(logits.cpu())
        all_y.append(batch["y"].cpu())
        all_risk.append(risk.cpu())
        all_time.append(batch["time"].cpu())
        all_event.append(event.cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    acc, f1 = accuracy_f1(logits, y)

    risk = torch.cat(all_risk, dim=0).numpy()
    time = torch.cat(all_time, dim=0).numpy()
    event = torch.cat(all_event, dim=0).numpy()
    ci = c_index(risk, time, event)

    return {
        "loss": float(np.mean(losses)) if len(losses) else 0.0,
        "acc": acc,
        "f1": f1,
        "cindex": ci,
    }

def main():
    import argparse
    from config import Config
    from model import MMDiffuzzy
    from utils import load_ckpt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config().ensure()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    paths = MultiModalPTDataset.discover(cfg.data_dir)
    ds = MultiModalPTDataset(cfg.data_dir, paths)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = MMDiffuzzy(cfg).to(device)
    load_ckpt(args.ckpt, model, optim=None, map_location=device)

    metrics = evaluate(model, loader, device)
    print(metrics)

if __name__ == "__main__":
    main()