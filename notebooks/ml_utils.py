import numpy as np

# ── train_test_split ──────────────────────────────────────────────────
def train_test_split(*arrays, test_size=0.25, random_state=None, stratify=None):
    """Split arrays into random train / test subsets (stratified optional)."""
    rng = np.random.RandomState(random_state)
    n = len(arrays[0])
    for a in arrays:
        assert len(a) == n, "All arrays must have the same length"

    if stratify is not None:
        train_idx, test_idx = [], []
        classes = np.unique(stratify)
        for c in classes:
            c_idx = np.where(stratify == c)[0]
            rng.shuffle(c_idx)
            n_test = max(1, int(round(len(c_idx) * test_size)))
            test_idx.extend(c_idx[:n_test])
            train_idx.extend(c_idx[n_test:])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    result = []
    for a in arrays:
        result.append(a[train_idx])
        result.append(a[test_idx])
    return result


# ── StandardScaler ────────────────────────────────────────────────────
class StandardScaler:
    """Z-score standardisation: (x - mean) / std."""
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0   # avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


# ── confusion_matrix ──────────────────────────────────────────────────
def confusion_matrix(y_true, y_pred):
    """Return a confusion matrix (classes sorted ascending)."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[c2i[t], c2i[p]] += 1
    return cm


# ── classification_report ─────────────────────────────────────────────
def classification_report(y_true, y_pred, target_names=None):
    """Return a text classification report (precision / recall / f1 / support)."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if target_names is None:
        target_names = [str(c) for c in classes]

    lines = []
    hdr = f"{'':>15s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}"
    lines.append(hdr)
    lines.append("")

    precisions, recalls, f1s, supports = [], [], [], []
    for c, name in zip(classes, target_names):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        sup = int(np.sum(y_true == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f); supports.append(sup)
        lines.append(f"{name:>15s} {p:10.2f} {r:10.2f} {f:10.2f} {sup:10d}")

    total = sum(supports)
    acc = np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0
    lines.append("")
    lines.append(f"{'accuracy':>15s} {'':>10s} {'':>10s} {acc:10.2f} {total:10d}")

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f = np.mean(f1s)
    lines.append(f"{'macro avg':>15s} {macro_p:10.2f} {macro_r:10.2f} {macro_f:10.2f} {total:10d}")

    w = np.array(supports, dtype=float)
    w_sum = w.sum() if w.sum() > 0 else 1.0
    wavg_p = np.dot(precisions, w) / w_sum
    wavg_r = np.dot(recalls, w) / w_sum
    wavg_f = np.dot(f1s, w) / w_sum
    lines.append(f"{'weighted avg':>15s} {wavg_p:10.2f} {wavg_r:10.2f} {wavg_f:10.2f} {total:10d}")
    lines.append("")

    return "\n".join(lines)
