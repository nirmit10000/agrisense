"""Train the AgriSense leaf-disease classifier (PlantVillage maize + tomato, 14 classes) on CPU.

Model: MobileNetV2 (ImageNet-pretrained) -> global average pool -> Linear(1280, 14).
Because the head is a single linear layer after global average pooling, the class activation map
(CAM) is exact: cam[c] = sum_k W[c,k] * features[k] and logits[c] = mean(cam[c]) + b[c].

    python leaf/train_leaf.py <data_dir>        # data_dir from leaf/fetch_data.py

Writes: web/models/leaf.onnx, web/data/leaf_classes.json, web/samples/, results/leaf_metrics.json,
results/fig_leaf_*.png
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WEB, RES = ROOT / "web", ROOT / "results"
SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# The website does exactly this to uploaded photos, so evaluation uses the same transform.
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
TRAIN_TF = T.Compose([T.Resize(256), T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(), T.ColorJitter(0.15, 0.15, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])

CLASS_INFO = {  # folder name -> (crop, display name, advice)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Maize", "Gray leaf spot",
        "Rotate away from maize, plough under infected residue and grow tolerant hybrids; avoid very dense "
        "planting so leaves dry faster. Confirm with your local Krishi Vigyan Kendra / agriculture officer "
        "before using any fungicide."),
    "Corn_(maize)___Common_rust_": (
        "Maize", "Common rust",
        "Usually minor on field maize: grow resistant hybrids and sow on time so the crop escapes peak rust "
        "weather."),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "Maize", "Northern leaf blight",
        "Use resistant hybrids, rotate crops and bury infected residue after harvest. Confirm with your local "
        "Krishi Vigyan Kendra / agriculture officer before using any fungicide."),
    "Corn_(maize)___healthy": (
        "Maize", "Healthy",
        "No disease detected. Keep scouting the crop weekly, especially after warm, humid spells, and confirm "
        "any new symptoms with your local Krishi Vigyan Kendra / agriculture officer before using chemicals."),
    "Tomato___Bacterial_spot": (
        "Tomato", "Bacterial spot",
        "Use certified disease-free seed and transplants, avoid overhead watering and remove badly infected "
        "leaves. Confirm with your local Krishi Vigyan Kendra / agriculture officer before using copper or "
        "other sprays."),
    "Tomato___Early_blight": (
        "Tomato", "Early blight",
        "Remove infected lower leaves, mulch so soil does not splash onto leaves, and rotate away from tomato "
        "and potato for 2-3 years. Confirm with your local Krishi Vigyan Kendra / agriculture officer before "
        "using any fungicide."),
    "Tomato___Late_blight": (
        "Tomato", "Late blight",
        "Spreads fast in cool, wet weather: remove and destroy infected plants promptly and keep foliage dry. "
        ""),
    "Tomato___Leaf_Mold": (
        "Tomato", "Leaf mold",
        "Mostly a problem in humid, enclosed conditions: improve ventilation, space plants wider and avoid "
        "wetting leaves. Confirm with your local Krishi Vigyan Kendra / agriculture officer before using any "
        "fungicide."),
    "Tomato___Septoria_leaf_spot": (
        "Tomato", "Septoria leaf spot",
        "Remove infected lower leaves, mulch, avoid overhead irrigation and rotate crops. Confirm with your "
        "local Krishi Vigyan Kendra / agriculture officer before using any fungicide."),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomato", "Spider mites",
        "A pest, not a disease, favoured by hot, dry, dusty conditions: spray leaves with water to dislodge "
        "mites and protect natural predators. Confirm with your local Krishi Vigyan Kendra / agriculture "
        "officer before using any miticide."),
    "Tomato___Target_Spot": (
        "Tomato", "Target spot",
        "Improve airflow, remove infected leaves and crop debris, and avoid long periods of leaf wetness. "
        ""),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato", "Yellow leaf curl virus",
        "Spread by whiteflies and cannot be cured: remove infected plants, control whiteflies with yellow "
        "sticky traps and insect-proof nursery nets, and grow resistant varieties. Confirm with your local "
        "Krishi Vigyan Kendra / agriculture officer before using any insecticide."),
    "Tomato___Tomato_mosaic_virus": (
        "Tomato", "Mosaic virus",
        "Spread by contact and on tools: remove infected plants, wash hands and disinfect tools, and grow "
        "resistant varieties. Confirm with your local Krishi Vigyan Kendra / agriculture officer before "
        "using any chemicals."),
    "Tomato___healthy": (
        "Tomato", "Healthy",
        "No disease detected. Keep scouting the crop weekly and confirm any new symptoms with your local "
        "Krishi Vigyan Kendra / agriculture officer before using chemicals."),
}


class LeafNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.features = base.features            # -> [B, 1280, 7, 7]
        self.head = nn.Linear(1280, n_classes)
        self.dropout = nn.Dropout(0.2)            # training only

    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.head(self.dropout(f))


class ExportNet(nn.Module):
    """Inference graph for the browser: returns logits and the class activation maps."""

    def __init__(self, net):
        super().__init__()
        self.features, self.head = net.features, net.head

    def forward(self, x):
        f = self.features(x)                                         # [1, 1280, 7, 7]
        cam = torch.einsum("ck,bkhw->bchw", self.head.weight, f)     # [1, C, 7, 7]
        logits = cam.mean(dim=(2, 3)) + self.head.bias               # identical to Linear(avgpool(f))
        return logits, cam


def load_split(data_dir: Path):
    color = data_dir / "pv" / "raw" / "color"
    keys = sorted(CLASS_INFO)
    items = []
    for cid, k in enumerate(keys):
        files = sorted((color / k).glob("*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        items += [(f, cid) for f in files]
    rng = random.Random(SEED)
    split = {"train": [], "val": [], "test": []}
    for cid in range(len(keys)):
        cls = [it for it in items if it[1] == cid]
        rng.shuffle(cls)
        n = len(cls)
        n_tr, n_va = int(round(0.70 * n)), int(round(0.15 * n))
        split["train"] += cls[:n_tr]
        split["val"] += cls[n_tr:n_tr + n_va]
        split["test"] += cls[n_tr + n_va:]
    return keys, split


def batches(items, tf, bs, shuffle=False, rng=None):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [items[j] for j in idx[i:i + bs]]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk])
        y = torch.tensor([c for _, c in chunk])
        yield x, y


@torch.no_grad()
def pooled_features(net, items, bs=64):
    net.eval()
    out, ys = [], []
    for x, y in batches(items, EVAL_TF, bs):
        out.append(F.adaptive_avg_pool2d(net.features(x), 1).flatten(1))
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


@torch.no_grad()
def predict(net, items, bs=64):
    net.eval()
    probs, ys = [], []
    for x, y in batches(items, EVAL_TF, bs):
        probs.append(F.softmax(net(x), 1))
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def main(data_dir: Path):
    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    keys, split = load_split(data_dir)
    C = len(keys)
    print({k: len(v) for k, v in split.items()}, "classes", C)
    net = LeafNet(C)
    history = []

    # ---- stage 1: linear head on frozen, cached features
    ftr, ytr = pooled_features(net, split["train"])
    fva, yva = pooled_features(net, split["val"])
    print(f"features cached ({time.time() - t0:.0f}s)")
    head_opt = torch.optim.Adam(net.head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    head_epochs = 40
    for ep in range(head_epochs):
        perm = torch.randperm(len(ftr), generator=g)
        for i in range(0, len(ftr), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(net.head(F.dropout(ftr[b], 0.2)), ytr[b])
            head_opt.zero_grad()
            loss.backward()
            head_opt.step()
        with torch.no_grad():
            acc = (net.head(fva).argmax(1) == yva).float().mean().item()
        history.append(dict(stage="head", epoch=ep + 1, val_acc=acc))
    print(f"head done: val acc {acc:.4f} ({time.time() - t0:.0f}s)")
    best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}

    # ---- stage 2: fine-tune the last inverted-residual blocks + head (time-boxed)
    budget_s = float(os.environ.get("FT_BUDGET_S", 1200))
    nb = int(os.environ.get("FT_BLOCKS", 4))          # how many of the last feature blocks to fine-tune
    max_ep = int(os.environ.get("FT_EPOCHS", 3))
    for p in net.features.parameters():
        p.requires_grad = False
    trainable = list(net.features[-nb:].parameters()) + list(net.head.parameters())
    for p in trainable:
        p.requires_grad = True
    ft_opt = torch.optim.Adam([{"params": list(net.features[-nb:].parameters()), "lr": 3e-4},
                               {"params": list(net.head.parameters()), "lr": 5e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(ft_opt, step_size=1, gamma=0.6)   # smaller steps each epoch
    rng = random.Random(SEED)
    ft_epochs, t_ft = 0, time.time()
    for ep in range(max_ep):
        net.train()
        net.features[:-nb].eval()   # keep frozen BatchNorm statistics fixed
        for x, y in batches(split["train"], TRAIN_TF, 32, shuffle=True, rng=rng):
            with torch.no_grad():
                h = net.features[:-nb](x)
            f = F.adaptive_avg_pool2d(net.features[-nb:](h), 1).flatten(1)
            loss = F.cross_entropy(net.head(net.dropout(f)), y)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()
        ft_epochs += 1
        sched.step()
        p_va, y_va = predict(net, split["val"])
        acc = float((p_va.argmax(1) == y_va).mean())
        history.append(dict(stage="finetune", epoch=ep + 1, val_acc=acc))
        print(f"finetune epoch {ep + 1}: val acc {acc:.4f} ({time.time() - t0:.0f}s)")
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        per_epoch = (time.time() - t_ft) / ft_epochs
        if time.time() - t_ft + per_epoch > budget_s:
            break
    net.load_state_dict(best_state)
    train_minutes = (time.time() - t0) / 60

    # ---- test evaluation
    p_te, y_te = predict(net, split["test"])
    pred = p_te.argmax(1)
    test_acc = float((pred == y_te).mean())
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_te, pred):
        cm[t, p] += 1
    per_class = []
    for c in range(C):
        tp = cm[c, c]
        prec = tp / max(1, cm[:, c].sum())
        rec = tp / max(1, cm[c].sum())
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        crop, name, _ = CLASS_INFO[keys[c]]
        per_class.append(dict(crop=crop, name=name, precision=round(float(prec), 4), recall=round(float(rec), 4),
                              f1=round(float(f1), 4), support=int(cm[c].sum())))
    macro_f1 = float(np.mean([pc["f1"] for pc in per_class]))
    print(f"TEST acc {test_acc:.4f} macro F1 {macro_f1:.4f}")

    # ---- export
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    onnx_path = WEB / "models" / "leaf.onnx"
    exp = ExportNet(net).eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(exp, dummy, str(onnx_path), input_names=["input"], output_names=["logits", "cam"],
                      opset_version=17, dynamo=False)
    verify_onnx(exp, onnx_path, split["test"][:5])
    torch.save(best_state, ROOT / "leaf" / "leaf_mobilenetv2.pt") if os.environ.get("SAVE_PT") else None

    classes = []
    for cid, k in enumerate(keys):
        crop, name, advice = CLASS_INFO[k]
        classes.append(dict(id=cid, key=k, crop=crop, name=name, healthy=name == "Healthy", advice=advice))
    (WEB / "data" / "leaf_classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    write_samples(keys, split["test"], pred, p_te)

    size_mb = onnx_path.stat().st_size / 1e6
    metrics = {
        "model": "MobileNetV2 (ImageNet-pretrained, fine-tuned)",
        "dataset": "PlantVillage (color), maize + tomato",
        "classes": [{k: c[k] for k in ("id", "crop", "name", "healthy")} for c in classes],
        "n_images": {k: len(v) for k, v in split.items()},
        "input_size": 224,
        "preprocess": "resize-short-256, center-crop-224, imagenet-norm",
        "epochs": {"head": head_epochs, "finetune": ft_epochs}, "finetune_blocks": nb,
        "train_minutes": round(train_minutes, 1),
        "val_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "model_file": "web/models/leaf.onnx",
        "model_size_mb": round(size_mb, 2),
    }
    RES.mkdir(exist_ok=True)
    (RES / "leaf_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figures(keys, split, cm, history, exp, p_te, y_te)
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "macro_f1", "train_minutes", "model_size_mb")}))


def verify_onnx(exp, path, items):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst, worst_id = 0.0, 0.0
    for p, _ in items:
        x = EVAL_TF(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            lt, ct = exp(x)
        lo, co = sess.run(["logits", "cam"], {"input": x.numpy()})
        worst = max(worst, float(np.abs(lo - lt.numpy()).max()), float(np.abs(co - ct.numpy()).max()))
        ident = co.mean(axis=(2, 3)) + exp.head.bias.detach().numpy()
        worst_id = max(worst_id, float(np.abs(ident - lo).max()))
    print(f"ONNX check: max|onnx-torch| = {worst:.2e}, max|mean(cam)+b - logits| = {worst_id:.2e}")
    assert worst < 1e-3 and worst_id < 1e-3


def slug(crop, name):
    return f"{crop}_{name}".lower().replace(" ", "_").replace("-", "_")


def write_samples(keys, test_items, pred, probs):
    out = WEB / "samples"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cid, k in enumerate(keys):
        crop, name, _ = CLASS_INFO[k]
        # correctly classified, confident test images make the clearest demo
        idx = [i for i, (_, c) in enumerate(test_items) if c == cid]
        idx.sort(key=lambda i: (pred[i] != cid, -probs[i][cid]))
        for n, i in enumerate(idx[:2], 1):
            img = Image.open(test_items[i][0]).convert("RGB")
            img.thumbnail((256, 256))
            fname = f"{slug(crop, name)}_{n}.jpg"
            img.save(out / fname, quality=85)
            manifest.append({"file": f"web/samples/{fname}", "label_id": cid})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def short(crop, name):
    return f"{'Mz.' if crop == 'Maize' else 'Tom.'} {name}"


def figures(keys, split, cm, history, exp, p_te, y_te):
    import matplotlib
    from agrisense.evaluate import BLUES, INK, INK2, POLICY_COLORS, SURFACE, _save  # applies the chart style
    import matplotlib.pyplot as plt
    labels = [short(*CLASS_INFO[k][:2]) for k in keys]
    C = len(keys)

    # confusion matrix (row-normalised)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7fbff"] + BLUES)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cmn):
        if v >= 0.005:
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(C), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(C), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    acc = float(np.trace(cm) / cm.sum())
    ax.set_title(f"Leaf disease classifier: test confusion matrix (% of each true class), "
                 f"accuracy {100 * acc:.1f}%", loc="left", fontsize=10)
    _save(fig, "fig_leaf_confusion.png")

    # CAM examples: 4 test images, raw and with heatmap
    rng = random.Random(7)
    chosen_classes = rng.sample(range(C), 4)
    test = split["test"]
    picks = []
    for c in chosen_classes:
        idx = [i for i, (_, cc) in enumerate(test) if cc == c]
        picks.append(idx[0])
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for col, i in enumerate(picks):
        path, true = test[i]
        img = Image.open(path).convert("RGB")
        x = EVAL_TF(img).unsqueeze(0)
        with torch.no_grad():
            logits, cam = exp(x)
        pc = int(logits.argmax())
        m = cam[0, pc].numpy()
        m = (m - m.min()) / max(1e-9, m.max() - m.min())
        crop_img = T.CenterCrop(224)(T.Resize(256)(img))
        heat = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255
        axes[0, col].imshow(crop_img)
        axes[0, col].set_title(f"true: {labels[true]}", fontsize=9, loc="left")
        axes[1, col].imshow(crop_img)
        axes[1, col].imshow(heat, cmap="inferno", alpha=0.5)
        prob = float(F.softmax(logits, 1)[0, pc])
        axes[1, col].set_title(f"pred: {labels[pc]} ({100 * prob:.0f}%)", fontsize=9, loc="left")
        for r in range(2):
            axes[r, col].axis("off")
    fig.suptitle("Where the model looks: class activation maps (bright = evidence for the predicted disease)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, "fig_leaf_cam_examples.png")

    # training curve
    fig, ax = plt.subplots(figsize=(7, 3.4))
    head = [h for h in history if h["stage"] == "head"]
    ft = [h for h in history if h["stage"] == "finetune"]
    ax.plot([h["epoch"] for h in head], [100 * h["val_acc"] for h in head], color=POLICY_COLORS["Q-learning agent (RL)"],
            label="stage 1: train classifier head (frozen backbone)")
    if ft:
        xs = [len(head) + h["epoch"] for h in ft]
        ax.plot(xs, [100 * h["val_acc"] for h in ft], color=POLICY_COLORS["Sensor threshold rule"], marker="o",
                label="stage 2: fine-tune last blocks")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Leaf classifier training", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig_leaf_training.png")

    # dataset
    counts = [sum(1 for _, c in split[s] if c == cid) for cid in range(C) for s in ("train",)]
    totals = [sum(1 for s in split for _, c in split[s] if c == cid) for cid in range(C)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#1baf7a" if CLASS_INFO[k][0] == "Maize" else "#eb6834" for k in keys]
    bars = ax.barh(labels, totals, color=colors, height=0.65)
    for b, v in zip(bars, totals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("Images per class used (PlantVillage, sampled; split 70/15/15)", loc="left")
    _save(fig, "fig_leaf_dataset.png")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
