"""Download a class-balanced sample of PlantVillage (color) maize + tomato images.

Uses a git partial clone + sparse checkout of github.com/spMohanty/PlantVillage-Dataset so only the
selected images are downloaded.

    python leaf/fetch_data.py <data_dir> [per_class=600]
"""
import collections
import random
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PREFIXES = ("raw/color/Corn_(maize)___", "raw/color/Tomato___")
SPECIAL = set("\\[]*?!#")


def escape(path: str) -> str:
    return "/" + "".join("\\" + ch if ch in SPECIAL else ch for ch in path)


def main(data_dir: Path, per_class: int = 600):
    data_dir.mkdir(parents=True, exist_ok=True)
    pv = data_dir / "pv"
    if not pv.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO, str(pv)],
                       check=True)
    subprocess.run(["git", "-C", str(pv), "sparse-checkout", "init", "--no-cone"], check=True)
    out = subprocess.run(["git", "-C", str(pv), "ls-tree", "-r", "--name-only", "HEAD", "--", "raw/color"],
                         check=True, capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    by_class = collections.defaultdict(list)
    for f in out:
        if f.startswith(PREFIXES):
            by_class[f.split("/")[2]].append(f)
    rng = random.Random(42)
    selected = []
    for k in sorted(by_class):
        files = sorted(by_class[k])
        rng.shuffle(files)
        selected += files[:per_class]
    (pv / ".git" / "info" / "sparse-checkout").write_text("\n".join(escape(p) for p in selected) + "\n",
                                                          encoding="utf-8")
    subprocess.run(["git", "-C", str(pv), "checkout"], check=True)
    print(f"{len(by_class)} classes, {len(selected)} images selected")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 600)
