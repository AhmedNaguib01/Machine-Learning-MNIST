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


# ── compute_accuracy ──────────────────────────────────────────────────
def compute_accuracy(y_true, y_pred):
    """Return the fraction of correct predictions."""
    return np.mean(y_true == y_pred)


# ── k_fold_split ──────────────────────────────────────────────────────
def k_fold_split(X, y, k=5, seed=42):
    """Split data into k folds and return list of (train_idx, val_idx) tuples."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    fold_sz = len(y) // k
    folds = []
    for i in range(k):
        s = i * fold_sz
        e = (i + 1) * fold_sz if i < k - 1 else len(y)
        v = idx[s:e]
        t = np.concatenate([idx[:s], idx[e:]])
        folds.append((t, v))
    return folds


# ── cross_validate ────────────────────────────────────────────────────
def cross_validate(model_cls, params, X, y, k=5):
    """Run k-fold cross-validation and return mean accuracy."""
    folds = k_fold_split(X, y, k=k)
    accs = []
    for fi, (ti, vi) in enumerate(folds):
        m = model_cls(**params)
        m.fit(X[ti], y[ti])
        acc = compute_accuracy(y[vi], m.predict(X[vi]))
        accs.append(acc)
        print(f"  Fold {fi+1}/{k}: {acc:.4f}")
    mean_acc = np.mean(accs)
    print(f"  Mean CV: {mean_acc:.4f}")
    return mean_acc


# ── plot_learning_curve ───────────────────────────────────────────────
def plot_learning_curve(model_cls, params, X_tr, y_tr, X_vl, y_vl, fracs, title):
    """Plot training & validation accuracy vs training set size."""
    import matplotlib.pyplot as plt

    tr_a, vl_a, szs = [], [], []
    n = len(y_tr)
    for f in fracs:
        sz = max(int(n * f), 10)
        szs.append(sz)
        idx = np.random.choice(n, sz, replace=False)
        m = model_cls(**params)
        m.fit(X_tr[idx], y_tr[idx])
        tr_a.append(compute_accuracy(y_tr[idx], m.predict(X_tr[idx])))
        vl_a.append(compute_accuracy(y_vl, m.predict(X_vl)))
    plt.figure(figsize=(8, 5))
    plt.plot(szs, tr_a, 'o-', label='Train', color='#2ecc71')
    plt.plot(szs, vl_a, 's-', label='Val', color='#e74c3c')
    plt.fill_between(szs, tr_a, vl_a, alpha=0.15, color='gray')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── evaluate_model ────────────────────────────────────────────────────
def evaluate_model(model, X_ts, y_ts, name):
    """Evaluate a trained model: print report and plot confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    preds = model.predict(X_ts)
    acc = compute_accuracy(y_ts, preds)
    classes = np.unique(np.concatenate([y_ts, preds]))
    target_names = [f'Digit {c}' for c in classes]

    print(f"\n{'='*70}")
    print(f"{name} — Test Acc: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*70}")
    print(classification_report(y_ts, preds, target_names=target_names))

    cm = confusion_matrix(y_ts, preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{name}\nTest Acc: {acc:.4f}', fontweight='bold')
    plt.tight_layout()
    plt.show()
    return acc

# ── show_misclassified ─── ─────────────────────────────────────────────────
def show_misclassified(y_true, y_pred, images, title='Misclassified Samples', n_show=10):
    """Show misclassified samples."""
    import matplotlib.pyplot as plt
    
    wrong = np.where(y_pred != y_true)[0]
    n_wrong = len(wrong)
    print(f'Misclassified: {n_wrong} / {len(y_true)}')
    n_show = min(n_wrong, n_show)
    if n_show == 0:
        print('No misclassified samples!')
        return
    cols = min(n_show, 5); rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3.5*rows))
    axes = np.array(axes).reshape(-1)
    for k in range(n_show):
        idx = wrong[k]
        axes[k].imshow(images[idx], cmap='gray')
        axes[k].set_title(f'T:{y_true[idx]} P:{y_pred[idx]}', color='red', fontsize=10)
        axes[k].axis('off')
    for k in range(n_show, len(axes)):
        axes[k].axis('off')
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()