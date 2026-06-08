# -*- coding: utf-8 -*-
"""
Первичный анализ текстового набора данных Sentiment Analysis Dataset.

Скрипт выполняет требования учебного ноутбука по обработке текста:
1) очистка текста;
2) лемматизация/нормализация токенов;
3) подсчёт частоты слов и построение графиков;
4) удаление стоп-слов;
5) TF-IDF-векторизация;
6) информационный поиск по корпусу.

Также скрипт формирует структуру датасета для задачи классификации:
    data/train.csv
    data/val.csv
    data/test.csv
    metadata/dataset_info.yaml
    metadata/metadata.csv
    metadata/split_counts.csv
    metadata/examples.csv

Использование:
    python analyze_sentiment_dataset.py --input_csv sentiment_data.csv --output_dir Sentiment_Analysis_NLP_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover
    nltk = None
    WordNetLemmatizer = None


RANDOM_STATE = 42
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
PALETTE = {
    "negative": "#d7191c",  # красный
    "neutral": "#66bd63",   # зелёный
    "positive": "#1a9641",  # тёмно-зелёный
}
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "to", "of", "in", "on", "for", "from", "by", "at", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "their", "our", "not", "no", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "may", "might",
    "must", "very", "so", "just", "also", "still", "one", "like", "get", "got", "go",
    "going", "much", "many", "really", "even", "use", "used", "want", "need", "make",
    "well", "see", "know", "time", "people", "thing", "things", "way", "say", "said",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Приводит текст к нижнему регистру, удаляет пунктуацию и лишние пробелы."""
    text = str(text).lower()
    text = text.replace("’", "'")
    # Сохраняем только латинские буквы и пробелы: датасет англоязычный.
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())
    return text


class Lemmatizer:
    """Лемматизатор с резервной реализацией, если корпус WordNet недоступен."""

    def __init__(self) -> None:
        self.mode = "rule_based"
        self.wordnet = None
        if WordNetLemmatizer is not None:
            try:
                # Проверяем доступность корпуса WordNet. Если он отсутствует,
                # пробуем скачать его стандартным способом NLTK. Если загрузка
                # невозможна, код остаётся работоспособным благодаря резервной
                # нормализации.
                import nltk.data  # noqa: F401
                try:
                    nltk.data.find("corpora/wordnet")
                except LookupError:
                    nltk.download("wordnet", quiet=True)
                    nltk.download("omw-1.4", quiet=True)
                nltk.data.find("corpora/wordnet")
                self.wordnet = WordNetLemmatizer()
                self.mode = "wordnet"
            except Exception:
                self.wordnet = None
                self.mode = "rule_based"

    @lru_cache(maxsize=200_000)
    def lemmatize_word(self, word: str) -> str:
        if not word:
            return word
        if self.wordnet is not None:
            return self.wordnet.lemmatize(word)
        # Простая резервная нормализация для англоязычных комментариев.
        # Она не заменяет полноценную лемматизацию, но уменьшает вариативность токенов.
        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 5 and word.endswith("ing"):
            return word[:-3]
        if len(word) > 4 and word.endswith("ed"):
            return word[:-2]
        if len(word) > 4 and word.endswith("es"):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def lemmatize_text(self, text: str) -> str:
        return " ".join(self.lemmatize_word(w) for w in text.split())


def remove_stopwords(text: str, stop_words: set[str]) -> str:
    return " ".join(w for w in text.split() if w not in stop_words)


def load_and_prepare(input_csv: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    raw = pd.read_csv(input_csv, encoding="utf-8")
    raw = raw.rename(columns={"Comment": "text", "Sentiment": "label"})
    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])
    initial_rows = len(raw)
    missing_text = int(raw["text"].isna().sum())
    raw = raw.dropna(subset=["text"]).copy()
    raw["text"] = raw["text"].astype(str).str.strip()
    raw = raw[raw["text"] != ""].copy()
    duplicate_text_label = int(raw.duplicated(["text", "label"]).sum())
    raw = raw.drop_duplicates(["text", "label"]).copy()

    # Удаляем тексты с противоречивыми метками, чтобы не было одного и того же
    # комментария с разными классами.
    label_counts = raw.groupby("text")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    conflicting_rows = int(raw[raw["text"].isin(conflicting_texts)].shape[0])
    clean = raw[~raw["text"].isin(conflicting_texts)].copy()
    clean["label"] = clean["label"].astype(int)
    clean["label_name"] = clean["label"].map(LABEL_MAP)
    # Отсекаем строки, которые после очистки не содержат латинских слов.
    tmp_clean = clean["text"].map(clean_text)
    empty_after_cleaning = int((tmp_clean == "").sum())
    clean = clean[tmp_clean != ""].copy()
    clean["id"] = np.arange(len(clean))

    stats = {
        "initial_rows": initial_rows,
        "missing_text_rows": missing_text,
        "duplicate_text_label_rows": duplicate_text_label,
        "conflicting_rows_removed": conflicting_rows,
        "empty_after_cleaning_rows": empty_after_cleaning,
        "final_rows": len(clean),
    }
    return clean[["id", "text", "label", "label_name"]], stats


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["label"],
        random_state=RANDOM_STATE,
    )
    return train.sort_values("id"), val.sort_values("id"), test.sort_values("id")


def add_text_processing_columns(df: pd.DataFrame, lemmatizer: Lemmatizer) -> pd.DataFrame:
    out = df.copy()
    out["clean_text"] = out["text"].map(clean_text)
    out["lemmas"] = out["clean_text"].map(lemmatizer.lemmatize_text)
    out["no_stopwords"] = out["lemmas"].map(lambda s: remove_stopwords(s, STOP_WORDS))
    out["word_count"] = out["clean_text"].map(lambda s: len(s.split()))
    out["char_count"] = out["clean_text"].map(len)
    return out


def save_dataset_structure(base_dir: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, stats: Dict[str, int]) -> None:
    data_dir = base_dir / "data"
    meta_dir = base_dir / "metadata"
    ensure_dir(data_dir)
    ensure_dir(meta_dir)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(data_dir / f"{name}.csv", index=False, encoding="utf-8")

    all_parts = []
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        tmp = part[["id", "label", "label_name", "word_count", "char_count"]].copy()
        tmp.insert(1, "split", split_name)
        all_parts.append(tmp)
    metadata = pd.concat(all_parts, ignore_index=True)
    metadata.to_csv(meta_dir / "metadata.csv", index=False, encoding="utf-8")

    split_counts = metadata.groupby(["split", "label_name"]).size().reset_index(name="count")
    split_counts.to_csv(meta_dir / "split_counts.csv", index=False, encoding="utf-8")

    examples = pd.concat([
        train.groupby("label_name", group_keys=False).head(2),
        val.groupby("label_name", group_keys=False).head(1),
    ])[ ["split" if "split" in train.columns else "id"] ] if False else None
    ex = train.groupby("label_name", group_keys=False).head(2)[["id", "text", "label", "label_name", "clean_text", "lemmas", "no_stopwords"]]
    ex.to_csv(meta_dir / "examples.csv", index=False, encoding="utf-8")

    lengths = pd.concat([train, val, test], ignore_index=True)
    dataset_info = {
        "name": "Sentiment Analysis Dataset",
        "source": "Kaggle: https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset",
        "task_type": "text classification / sentiment analysis",
        "language": "English",
        "format": "CSV, UTF-8",
        "classes": LABEL_MAP,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "rows_before_cleaning": stats["initial_rows"],
        "rows_after_cleaning": stats["final_rows"],
        "removed": {
            "missing_text_rows": stats["missing_text_rows"],
            "duplicate_text_label_rows": stats["duplicate_text_label_rows"],
            "conflicting_rows": stats["conflicting_rows_removed"],
            "empty_after_cleaning_rows": stats["empty_after_cleaning_rows"],
        },
        "average_word_count": round(float(lengths["word_count"].mean()), 2),
        "median_word_count": round(float(lengths["word_count"].median()), 2),
        "date_prepared": "2026-05-24",
        "license": "CC0 1.0 Universal (Public Domain)",
    }
    if yaml is not None:
        with open(meta_dir / "dataset_info.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_info, f, allow_unicode=True, sort_keys=False)
    else:
        with open(meta_dir / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    dataset_card = f"""# Sentiment Analysis Dataset\n\nИсточник: Kaggle — https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset\n\nТип задачи: классификация тональности текста.\n\nКлассы: 0 — negative, 1 — neutral, 2 — positive.\n\nСтруктура: `data/train.csv`, `data/val.csv`, `data/test.csv`.\n\nКодировка: UTF-8.\n\nИтоговый размер после удаления пропусков, дубликатов и противоречивых меток: {stats['final_rows']} записей.\n"""
    with open(meta_dir / "dataset_card.md", "w", encoding="utf-8") as f:
        f.write(dataset_card)


def red_green_wordcloud_color(*args, **kwargs) -> str:
    shades = ["#d7191c", "#a50026", "#1a9641", "#006837", "#66bd63"]
    return random.choice(shades)


def build_figures(df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    ensure_dir(output_dir)
    figures = {}

    # 1. Баланс классов
    counts = df["label_name"].value_counts().reindex(["negative", "neutral", "positive"])
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=[PALETTE[i] for i in counts.index], edgecolor="black", linewidth=0.6)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{int(b.get_height()):,}".replace(',', ' '), ha="center", va="bottom", fontsize=10)
    plt.title("Распределение классов тональности")
    plt.xlabel("Класс")
    plt.ylabel("Количество комментариев")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figures["class_distribution"] = output_dir / "fig1_class_distribution.png"
    plt.savefig(figures["class_distribution"], dpi=180)
    plt.close()

    # 2. Распределение длин текстов
    plt.figure(figsize=(9, 5))
    bins = np.arange(0, min(160, int(df["word_count"].quantile(0.99)) + 10), 5)
    for label in ["negative", "neutral", "positive"]:
        subset = df.loc[df["label_name"] == label, "word_count"]
        plt.hist(subset, bins=bins, alpha=0.42, label=label, color=PALETTE[label], edgecolor="black", linewidth=0.2)
    plt.title("Распределение длины комментариев по классам")
    plt.xlabel("Длина текста, слов")
    plt.ylabel("Количество комментариев")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figures["length_distribution"] = output_dir / "fig2_length_distribution.png"
    plt.savefig(figures["length_distribution"], dpi=180)
    plt.close()

    # 3. Top-10 слов до удаления стоп-слов
    words = " ".join(df["lemmas"].astype(str)).split()
    top = Counter(words).most_common(10)
    top_words, top_counts = zip(*top)
    plt.figure(figsize=(9, 5))
    plt.bar(top_words, top_counts, color="#d7191c", edgecolor="black", linewidth=0.5)
    plt.title("Топ-10 слов до удаления стоп-слов")
    plt.xlabel("Слово")
    plt.ylabel("Частота")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figures["top_words_before_stop"] = output_dir / "fig3_top_words_before_stop.png"
    plt.savefig(figures["top_words_before_stop"], dpi=180)
    plt.close()

    # 4. WordCloud после лемматизации
    # Ограничиваем объём текста для устойчивой генерации.
    sample_text = " ".join(df["no_stopwords"].sample(min(50000, len(df)), random_state=RANDOM_STATE).astype(str))
    wc = WordCloud(width=1200, height=600, max_words=100, background_color="white", color_func=red_green_wordcloud_color, collocations=False).generate(sample_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Облако слов после удаления стоп-слов")
    plt.tight_layout()
    figures["wordcloud"] = output_dir / "fig4_wordcloud.png"
    plt.savefig(figures["wordcloud"], dpi=180)
    plt.close()

    # 5. Top-10 после удаления стоп-слов
    words_no = " ".join(df["no_stopwords"].astype(str)).split()
    top_no = Counter(words_no).most_common(10)
    top_words_no, top_counts_no = zip(*top_no)
    plt.figure(figsize=(9, 5))
    plt.bar(top_words_no, top_counts_no, color="#1a9641", edgecolor="black", linewidth=0.5)
    plt.title("Топ-10 слов после удаления стоп-слов")
    plt.xlabel("Слово")
    plt.ylabel("Частота")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figures["top_words_after_stop"] = output_dir / "fig5_top_words_after_stop.png"
    plt.savefig(figures["top_words_after_stop"], dpi=180)
    plt.close()

    # 6. Распределение классов по split
    if "split" in df.columns:
        split_counts = df.groupby(["split", "label_name"]).size().unstack(fill_value=0).reindex(columns=["negative", "neutral", "positive"])
        split_counts.plot(kind="bar", figsize=(9, 5), color=[PALETTE[c] for c in split_counts.columns], edgecolor="black", linewidth=0.5)
        plt.title("Сохранение баланса классов в train/val/test")
        plt.xlabel("Выборка")
        plt.ylabel("Количество комментариев")
        plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        figures["split_distribution"] = output_dir / "fig6_split_distribution.png"
        plt.savefig(figures["split_distribution"], dpi=180)
        plt.close()

    return figures


def build_tfidf_and_search(df: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["no_stopwords"].astype(str))
    feature_names = vectorizer.get_feature_names_out()

    first_vec = tfidf_matrix[0].toarray()[0]
    nonzero = np.where(first_vec > 0)[0]
    top_vector_items = sorted([(feature_names[i], float(first_vec[i])) for i in nonzero], key=lambda x: x[1], reverse=True)[:10]

    def search_texts(query: str, top_n: int = 3) -> List[Dict[str, object]]:
        q = remove_stopwords(Lemmatizer().lemmatize_text(clean_text(query)), STOP_WORDS)
        q_vec = vectorizer.transform([q])
        sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        idx = sims.argsort()[-top_n:][::-1]
        results = []
        for rank, i in enumerate(idx, 1):
            results.append({
                "query": query,
                "rank": rank,
                "similarity": float(sims[i]),
                "label": int(df.iloc[i]["label"]),
                "label_name": str(df.iloc[i]["label_name"]),
                "text": str(df.iloc[i]["text"]),
            })
        return results

    queries = ["apple pay secure convenient", "bad quality problem", "great helpful video"]
    search_results = []
    for q in queries:
        search_results.extend(search_texts(q, top_n=3))

    pd.DataFrame(search_results).to_csv(output_dir / "search_results.csv", index=False, encoding="utf-8")
    with open(output_dir / "tfidf_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocabulary_size": int(len(feature_names)),
            "matrix_shape": list(tfidf_matrix.shape),
            "first_text_top_vector_items": top_vector_items,
        }, f, ensure_ascii=False, indent=2)
    return {
        "vocabulary_size": int(len(feature_names)),
        "matrix_shape": tfidf_matrix.shape,
        "first_text_top_vector_items": top_vector_items,
        "search_results": search_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Путь к sentiment_data.csv")
    parser.add_argument("--output_dir", required=True, help="Папка для структурированного датасета и результатов")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    base_dir = Path(args.output_dir)
    figures_dir = base_dir / "figures"
    analysis_dir = base_dir / "analysis"
    ensure_dir(base_dir)
    ensure_dir(figures_dir)
    ensure_dir(analysis_dir)

    clean_df, stats = load_and_prepare(input_csv)
    lemmatizer = Lemmatizer()
    clean_df = add_text_processing_columns(clean_df, lemmatizer)

    train, val, test = split_dataset(clean_df)
    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")
    all_splits = pd.concat([train, val, test], ignore_index=True)

    save_dataset_structure(base_dir, train, val, test, stats)
    figures = build_figures(all_splits, figures_dir)
    tfidf_info = build_tfidf_and_search(all_splits, analysis_dir)

    # Сохраняем краткую сводку для отчёта.
    summary = {
        "cleaning_stats": stats,
        "label_distribution": all_splits["label_name"].value_counts().to_dict(),
        "split_sizes": all_splits["split"].value_counts().to_dict(),
        "length_stats": {
            "mean_words": round(float(all_splits["word_count"].mean()), 2),
            "median_words": round(float(all_splits["word_count"].median()), 2),
            "min_words": int(all_splits["word_count"].min()),
            "max_words": int(all_splits["word_count"].max()),
        },
        "lemmatizer_mode": lemmatizer.mode,
        "tfidf": {
            "vocabulary_size": tfidf_info["vocabulary_size"],
            "matrix_shape": list(tfidf_info["matrix_shape"]),
        },
        "figures": {k: str(v) for k, v in figures.items()},
    }
    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Готово.")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
