import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = train + val + test
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Les ratios doivent sommer à 1.0 (actuel: {s})")
    return train, val, test


def split_list(items: List[Any], ratios: Tuple[float, float, float]):
    train_r, val_r, test_r = ratios
    n = len(items)
    n_train = int(train_r * n)
    n_val = int(val_r * n)
    return (
        items[:n_train],
        items[n_train:n_train + n_val],
        items[n_train + n_val:]
    )


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


IMG_EXT = ".png"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    txt_path: Path


def find_image_for_label(label_path: Path, images_dir: Path) -> Path:
    stem = label_path.stem
    img = images_dir / f"{stem}{IMG_EXT}"
    if img.exists():
        return img

    raise FileNotFoundError(f"Aucune image trouvée pour {label_path.name}")


def load_samples(images_dir: str, labels_dir: str) -> List[Sample]:
    images_p = Path(images_dir)
    labels_p = Path(labels_dir)

    txt_files = sorted(labels_p.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Aucun .txt trouvé dans {labels_p}")

    samples: List[Sample] = []
    for txt in txt_files:
        img = find_image_for_label(txt, images_p)
        samples.append(Sample(image_path=img, txt_path=txt))

    return samples


def yolo_line_to_coco_bbox(line: str, img_w: int, img_h: int):
    parts = line.strip().split()
    cls = int(float(parts[0]))
    xc = float(parts[1]) * img_w
    yc = float(parts[2]) * img_h
    bw = float(parts[3]) * img_w
    bh = float(parts[4]) * img_h

    x = xc - bw / 2
    y = yc - bh / 2

    return cls, [x, y, bw, bh]


def build_coco_from_samples(
    samples: List[Sample],
    force_single_category_name: Optional[str] = None,
):
    images = []
    annotations = []

    if force_single_category_name:
        categories = [{
            "id": 0,
            "name": force_single_category_name,
            "supercategory": "none"
        }]
        single_class = True
    else:
        categories_map = {}
        single_class = False

    ann_id = 1

    for img_id, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as im:
            w, h = im.size

        images.append({
            "id": img_id,
            "file_name": sample.image_path.name,
            "width": w,
            "height": h,
        })

        lines = sample.txt_path.read_text().strip().splitlines()

        for line in lines:
            if not line.strip():
                continue

            cls, bbox = yolo_line_to_coco_bbox(line, w, h)
            cat_id = 0 if single_class else cls

            if not single_class:
                categories_map.setdefault(cls, {
                    "id": cls,
                    "name": f"class_{cls}",
                    "supercategory": "none"
                })

            area = bbox[2] * bbox[3]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    if not single_class:
        categories = [categories_map[k] for k in sorted(categories_map.keys())]

    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write_split(
    name: str,
    samples: List[Sample],
    output_root: Path,
    force_single_category_name: Optional[str] = None,
):
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copier image + txt dans le même dossier
    for s in samples:
        copy_file(s.image_path, out_dir / s.image_path.name)
        copy_file(s.txt_path, out_dir / s.txt_path.name)

    coco = build_coco_from_samples(
        samples,
        force_single_category_name=force_single_category_name
    )

    save_json(coco, out_dir / "_annotations.coco.json")

    print(f"✅ {name}: {len(samples)} images | {len(coco['annotations'])} annotations")


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_root: str,
    ratios=(0.8, 0.1, 0.1),
    seed=42,
    force_single_category_name: Optional[str] = None,
):
    ratios = normalize_ratios(*ratios)

    samples = load_samples(images_dir, labels_dir)

    random.seed(seed)
    random.shuffle(samples)

    train_s, val_s, test_s = split_list(samples, ratios)

    output_root = Path(output_root)

    write_split("train", train_s, output_root, force_single_category_name)
    write_split("valid", val_s, output_root, force_single_category_name)
    write_split("test", test_s, output_root, force_single_category_name)


if __name__ == "__main__":
    split_dataset(
        images_dir="../../../Dataset2D/ImagesEntrainement/images/train/",
        labels_dir="../../../Dataset2D/ImagesEntrainement/labels/train/",
        output_root="../../../../Dataset2D/COCO_Dataset",
        ratios=(0.8, 0.1, 0.1),
        seed=42,

    )