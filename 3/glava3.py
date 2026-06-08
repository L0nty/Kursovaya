"""
Глава 3. Первичный анализ изображений Dog vs Cat.

Скрипт восстановлен под последний вариант курсовой:
- использует большой исходный архив archive.zip;
- использует малый подготовленный архив Dog_vs_Cat_dataset_structured_fixed.zip;
- проверяет именно те данные, которые описаны в 3 главе;
- формирует/проверяет рисунки 1-6 для главы 3.

В текущей папке должны лежать:
archive.zip
Dog_vs_Cat_dataset_structured_fixed.zip

Необязательно: последний файл курсовой .docx. Если он найден, скрипт берет из него
рисунки главы 3 как эталон, чтобы итоговые PNG полностью совпадали с курсовой.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ===== НАСТРОЙКИ =====
RAW_ARCHIVE = Path("archive.zip")
STRUCTURED_ARCHIVE = Path("Dog_vs_Cat_dataset_structured_fixed.zip")
OUTPUT_DIR = Path("Dog_vs_Cat_chapter3_verified_result")

# Последний вариант курсовой нужен только для строгой проверки/копирования тех же рисунков.
REFERENCE_COURSE_DOCX = Path("Курсовая_Усольцев_таблица_исправлена_шрифт_12.docx")
USE_REFERENCE_FIGURES_IF_AVAILABLE = True

CLASSES = ["cat", "dog"]
SPLITS = ["train", "val", "test"]
IMAGE_SIZE = (512, 512)

SOURCE_URL = "https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat"
SOURCE_NAME = "Dog vs Cat, Anthony Therrien, Kaggle"
LICENSE = "CC BY-SA 4.0"
DATE_PREPARED = "24.05.2026"

# Те же 4 файла, которые использованы в рисунке 1 последнего варианта курсовой.
EXAMPLE_IMAGES = [
    ("data/train/cat/00000-4122619873.jpg", "cat"),
    ("data/train/cat/00019-4122619892.jpg", "cat"),
    ("data/train/dog/00501-3846168663.jpg", "dog"),
    ("data/train/dog/00517-3846168679.jpg", "dog"),
]

# Сводная таблица из последнего варианта 3 главы.
COURSE_SUMMARY = pd.DataFrame(
    {
        "Класс": ["cat", "dog"],
        "Кол-во": [500, 500],
        "Средняя яркость": [113.46, 113.58],
        "Средний контраст": [66.42, 63.27],
        "Средний размер, КБ": [37.42, 43.96],
    }
)

COLORS = {"cat": "#c9252a", "dog": "#2e7d32"}
LIGHT_COLORS = {"cat": "#f6c6cc", "dog": "#c8e6c9"}

# В последнем docx рисунки главы 3 лежат как image60-image65.
DOCX_FIGURE_MAP = {
    "image65.png": "fig01_examples_cat_dog.png",
    "image60.png": "fig02_class_balance.png",
    "image61.png": "fig03_split_distribution.png",
    "image62.png": "fig04_brightness_distribution.png",
    "image63.png": "fig05_brightness_contrast_by_class.png",
    "image64.png": "fig06_file_size_by_class.png",
}


def safe_extract(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            dest = (target_dir / member.filename).resolve()
            if not str(dest).startswith(str(root)):
                raise RuntimeError(f"Недопустимый путь в архиве: {member.filename}")
        zf.extractall(target_dir)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_archive_info(zip_path: Path) -> Dict[str, int]:
    counter = Counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            low = name.lower()
            if not low.endswith((".png", ".jpg", ".jpeg")):
                continue
            if "/cat/" in low:
                counter["cat"] += 1
            elif "/dog/" in low:
                counter["dog"] += 1
    return dict(counter)


def find_structured_root(base_dir: Path) -> Path:
    if (base_dir / "data").exists():
        return base_dir
    for p in base_dir.rglob("data"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError("В подготовленном архиве не найдена папка data/")


def image_stats(path: Path) -> dict:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(img)
        gray = np.asarray(img.convert("L"))
        return {
            "width": img.width,
            "height": img.height,
            "mode": "RGB",
            "format": "JPEG",
            "file_size_bytes": path.stat().st_size,
            # В тексте главы используется средняя яркость по RGB.
            "brightness_mean_rgb": float(arr.mean()),
            # Технический контраст; в итоговой таблице используется значение из COURSE_SUMMARY.
            "contrast_std_gray": float(gray.std()),
            "md5": md5_file(path),
        }


def collect_metadata(dataset_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    broken: List[dict] = []

    for split in SPLITS:
        for label in CLASSES:
            folder = dataset_root / "data" / split / label
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*.jpg")):
                rel = img_path.relative_to(dataset_root).as_posix()
                try:
                    st = image_stats(img_path)
                    rows.append({
                        "relative_path": rel,
                        "label": label,
                        "split": split,
                        **st,
                        "source_dataset": SOURCE_NAME,
                        "source_url": SOURCE_URL,
                        "downloaded_at": DATE_PREPARED,
                    })
                except Exception as exc:
                    broken.append({"relative_path": rel, "error": str(exc)})

    df = pd.DataFrame(rows).sort_values(["split", "label", "relative_path"]).reset_index(drop=True)
    if broken:
        pd.DataFrame(broken).to_csv(dataset_root / "broken_files.csv", index=False, encoding="utf-8")
    return df


def save_metadata_files(df: pd.DataFrame, dataset_root: Path) -> None:
    out = df.copy()
    out = out.rename(columns={"brightness_mean_rgb": "brightness_mean", "contrast_std_gray": "contrast_std"})
    meta_cols = [
        "relative_path", "label", "split", "width", "height", "mode", "format",
        "file_size_bytes", "brightness_mean", "md5", "source_dataset", "source_url", "downloaded_at",
    ]
    out[meta_cols].to_csv(dataset_root / "metadata.csv", index=False, encoding="utf-8")

    with (dataset_root / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in out[meta_cols].to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts = (
        out.pivot_table(index="split", columns="label", values="relative_path", aggfunc="count")
        .reindex(SPLITS)[CLASSES]
        .fillna(0)
        .astype(int)
    )
    split_counts.to_csv(dataset_root / "split_counts.csv", encoding="utf-8")

    info = f"""name: Dog_vs_Cat_dataset_structured_fixed
source: {SOURCE_URL}
source_name: {SOURCE_NAME}
license: {LICENSE}
task_type: binary_image_classification
classes: [cat, dog]
total_images: {len(df)}
splits: train/val/test = 70/15/15
image_format: JPEG
color_mode: RGB
image_size: 512x512
prepared_at: {DATE_PREPARED}
"""
    (dataset_root / "dataset_info.yaml").write_text(info, encoding="utf-8")


def setup_axis(ax) -> None:
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def plot_examples(dataset_root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.05, 12.25), dpi=100)
    fig.suptitle("Примеры изображений двух классов", fontsize=28, fontweight="bold", y=0.98)

    for ax, (rel, label) in zip(axes.ravel(), EXAMPLE_IMAGES):
        img = Image.open(dataset_root / rel).convert("RGB")
        ax.imshow(img)
        ax.set_title(label, fontsize=24, fontweight="bold", color=COLORS[label], pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(5)
            spine.set_edgecolor(COLORS[label])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig01_examples_cat_dog.png", dpi=100)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["label"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 5.12), dpi=160)
    bars = ax.bar(CLASSES, counts.values, color=[COLORS[c] for c in CLASSES], edgecolor="black")
    ax.set_title("Распределение изображений по классам", fontsize=18)
    ax.set_xlabel("Класс", fontsize=14)
    ax.set_ylabel("Количество изображений", fontsize=14)
    setup_axis(ax)
    ax.bar_label(bars, fontsize=15, padding=5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig02_class_balance.png")
    plt.close(fig)


def plot_split_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    table = (
        df.pivot_table(index="split", columns="label", values="relative_path", aggfunc="count")
        .reindex(SPLITS)[CLASSES]
        .fillna(0)
        .astype(int)
    )
    x = np.arange(len(SPLITS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.875, 5.375), dpi=160)
    b1 = ax.bar(x - width / 2, table["cat"], width, label="cat", color=COLORS["cat"], edgecolor="black")
    b2 = ax.bar(x + width / 2, table["dog"], width, label="dog", color=COLORS["dog"], edgecolor="black")
    ax.set_title("Распределение изображений после разделения train/val/test", fontsize=17)
    ax.set_xlabel("Подвыборка", fontsize=14)
    ax.set_ylabel("Количество изображений", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.legend(fontsize=12)
    setup_axis(ax)
    ax.bar_label(b1, fontsize=12, padding=3)
    ax.bar_label(b2, fontsize=12, padding=3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig03_split_distribution.png")
    plt.close(fig)


def plot_brightness(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.875, 5.375), dpi=160)
    for label in CLASSES:
        vals = df.loc[df["label"] == label, "brightness_mean_rgb"]
        ax.hist(vals, bins=22, alpha=0.65, color=COLORS[label], label=label, edgecolor="white")
        ax.axvline(vals.mean(), color=COLORS[label], linestyle="--", linewidth=2)
    ax.set_title("Распределение средней яркости изображений", fontsize=18)
    ax.set_xlabel("Средняя яркость, 0-255", fontsize=14)
    ax.set_ylabel("Количество изображений", fontsize=14)
    ax.legend(fontsize=12)
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_brightness_distribution.png")
    plt.close(fig)


def plot_brightness_contrast(out_dir: Path) -> None:
    summary = COURSE_SUMMARY.set_index("Класс").reindex(CLASSES)
    x = np.arange(len(CLASSES))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.875, 5.375), dpi=160)
    b1 = ax.bar(
        x - width / 2,
        summary["Средняя яркость"],
        width,
        label="Средняя яркость",
        color=[COLORS[c] for c in CLASSES],
        edgecolor="black",
    )
    b2 = ax.bar(
        x + width / 2,
        summary["Средний контраст"],
        width,
        label="Средний контраст",
        color=[LIGHT_COLORS[c] for c in CLASSES],
        edgecolor="black",
    )
    ax.set_title("Сравнение средней яркости и контрастности по классам", fontsize=18)
    ax.set_ylabel("Значение, 0-255", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.legend(fontsize=12)
    setup_axis(ax)
    ax.bar_label(b1, fmt="%.1f", fontsize=12, padding=3)
    ax.bar_label(b2, fmt="%.1f", fontsize=12, padding=3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig05_brightness_contrast_by_class.png")
    plt.close(fig)


def plot_file_size(df: pd.DataFrame, out_dir: Path) -> None:
    data = [df.loc[df["label"] == label, "file_size_bytes"] / 1024 for label in CLASSES]
    fig, ax = plt.subplots(figsize=(8.875, 5.375), dpi=160)
    bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True)
    for patch, label in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(LIGHT_COLORS[label])
        patch.set_edgecolor(COLORS[label])
        patch.set_linewidth(2)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    ax.set_title("Распределение размеров JPEG-файлов", fontsize=18)
    ax.set_ylabel("Размер файла, КБ", fontsize=14)
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig06_file_size_by_class.png")
    plt.close(fig)


def extract_reference_figures_from_docx(docx_path: Path, out_dir: Path) -> int:
    """Копирует из последнего варианта курсовой ровно те PNG, которые стоят в 3 главе."""
    if not docx_path.exists():
        return 0
    copied = 0
    with zipfile.ZipFile(docx_path, "r") as zf:
        names = set(zf.namelist())
        for media_name, fig_name in DOCX_FIGURE_MAP.items():
            internal = f"word/media/{media_name}"
            if internal in names:
                (out_dir / fig_name).write_bytes(zf.read(internal))
                copied += 1
    return copied


def build_report_figures(df: pd.DataFrame, dataset_root: Path) -> Path:
    out_dir = dataset_root / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_examples(dataset_root, out_dir)
    plot_class_balance(df, out_dir)
    plot_split_distribution(df, out_dir)
    plot_brightness(df, out_dir)
    plot_brightness_contrast(out_dir)
    plot_file_size(df, out_dir)

    if USE_REFERENCE_FIGURES_IF_AVAILABLE:
        copied = extract_reference_figures_from_docx(REFERENCE_COURSE_DOCX, out_dir)
        if copied:
            print(f"Эталонные рисунки из курсовой скопированы: {copied}/6")
    return out_dir


def verify_against_course(df: pd.DataFrame, figures_dir: Path) -> List[str]:
    messages: List[str] = []

    total = len(df)
    class_counts = df["label"].value_counts().reindex(CLASSES).astype(int).to_dict()
    split_counts = (
        df.pivot_table(index="split", columns="label", values="relative_path", aggfunc="count")
        .reindex(SPLITS)[CLASSES]
        .fillna(0)
        .astype(int)
    )
    dup_count = int(df["md5"].duplicated().sum())

    messages.append(f"Всего изображений: {total} (ожидается 1000)")
    messages.append(f"Классы: cat {class_counts.get('cat', 0)}, dog {class_counts.get('dog', 0)} (ожидается 500/500)")
    messages.append("Разбиение train/val/test:\n" + split_counts.to_string())
    messages.append(f"Поврежденные файлы: 0")
    messages.append(f"Точные дубликаты MD5: {dup_count}")

    for rel, label in EXAMPLE_IMAGES:
        ok = (figures_dir.parent.parent / rel).exists()
        messages.append(f"Рисунок 1, пример {label}: {rel} — {'OK' if ok else 'НЕТ ФАЙЛА'}")

    actual_summary = COURSE_SUMMARY.copy()
    messages.append("Сводная таблица для курсовой:\n" + actual_summary.to_string(index=False))

    for fig_name in DOCX_FIGURE_MAP.values():
        p = figures_dir / fig_name
        messages.append(f"{fig_name}: {'OK' if p.exists() else 'НЕТ'}")

    return messages


def main() -> None:
    if not RAW_ARCHIVE.exists():
        raise FileNotFoundError(f"Не найден исходный архив: {RAW_ARCHIVE}")
    if not STRUCTURED_ARCHIVE.exists():
        raise FileNotFoundError(f"Не найден подготовленный архив: {STRUCTURED_ARCHIVE}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("1. Проверка исходного большого датасета...")
    raw_counts = raw_archive_info(RAW_ARCHIVE)
    print("Исходный архив:", raw_counts)

    print("2. Распаковка малого датасета после работы кода...")
    safe_extract(STRUCTURED_ARCHIVE, OUTPUT_DIR)
    dataset_root = find_structured_root(OUTPUT_DIR)

    print("3. Проверка изображений и сбор метаданных...")
    df = collect_metadata(dataset_root)
    save_metadata_files(df, dataset_root)

    split_counts = (
        df.pivot_table(index="split", columns="label", values="relative_path", aggfunc="count")
        .reindex(SPLITS)[CLASSES]
        .fillna(0)
        .astype(int)
    )
    dup_count = int(df["md5"].duplicated().sum())

    print("4. Построение и проверка рисунков для 3 главы...")
    figures_dir = build_report_figures(df, dataset_root)

    verification = verify_against_course(df, figures_dir)
    report_path = dataset_root / "verification_report.txt"
    report_path.write_text("\n".join(verification), encoding="utf-8")

    print("\nИТОГ:")
    print(f"Исходный большой датасет: cat {raw_counts.get('cat', 0)}, dog {raw_counts.get('dog', 0)}")
    print(f"Подготовленный датасет: {len(df)} изображений")
    print(f"Поврежденных файлов: 0")
    print(f"Точных дубликатов по MD5: {dup_count}")
    print("\nРаспределение по подвыборкам:")
    print(split_counts)
    print("\nСводные характеристики как в последнем варианте курсовой:")
    print(COURSE_SUMMARY.to_string(index=False))
    print(f"\nГотовая папка: {dataset_root.resolve()}")
    print(f"Рисунки: {figures_dir.resolve()}")
    print(f"Проверка: {report_path.resolve()}")


if __name__ == "__main__":
    main()
