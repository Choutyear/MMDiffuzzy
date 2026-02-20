import numpy as np
import torch

def accuracy_f1(logits: torch.Tensor, y: torch.Tensor):
    pred = torch.argmax(logits, dim=1)
    y = y.view(-1)
    pred = pred.view(-1)
    acc = (pred == y).float().mean().item()

    num_classes = int(torch.max(y).item()) + 1
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    f1_macro = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    return acc, f1_macro

def c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray):
    """
    Concordance index.
    event: 1 means event observed (NOT censored), 0 means censored.
    """
    risk = np.asarray(risk).reshape(-1)
    time = np.asarray(time).reshape(-1)
    event = np.asarray(event).reshape(-1)

    n = len(time)
    concordant = 0.0
    comparable = 0.0
    tied = 0.0
    for i in range(n):
        if event[i] <= 0.5:
            continue
        for j in range(n):
            if time[i] < time[j]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    tied += 1.0
    if comparable == 0:
        return 0.0
    return float((concordant + 0.5 * tied) / comparable)