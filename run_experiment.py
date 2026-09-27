"""Experiment comparing a classic text classifier with Gemini API.

The script is intentionally small and linear so that each step can also be
followed from the accompanying notebook.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "customer_reviews_sentiment.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCUMENTATION_DIR = PROJECT_ROOT / "documentation"
LABELS = ["negatif", "positif"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the original dataset without modifying its content."""

    data = pd.read_csv(data_path)
    required_columns = {"review_id", "product_name", "review_text", "sentiment"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Kolom dataset belum lengkap: {sorted(missing_columns)}")

    data = data.dropna(subset=["review_text", "sentiment"]).copy()
    data["sentiment"] = data["sentiment"].str.lower().str.strip()
    invalid_labels = set(data["sentiment"]) - set(LABELS)
    if invalid_labels:
        raise ValueError(f"Label tidak dikenali: {sorted(invalid_labels)}")
    return data


def split_dataset(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one stratified split shared by both approaches."""

    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["sentiment"],
    )
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def train_classic_model(train_data: pd.DataFrame) -> Pipeline:
    """Train a simple TF-IDF + Logistic Regression text classifier."""

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )
    model.fit(train_data["review_text"], train_data["sentiment"])
    return model


def build_gemini_prompt(review: str) -> str:
    """Build a constrained few-shot prompt for one review."""

    return f"""Anda adalah classifier sentiment untuk ulasan pelanggan e-commerce.

Tentukan satu label saja berdasarkan teks ulasan:
- positif: pelanggan puas atau menyampaikan pengalaman baik.
- negatif: pelanggan kecewa, menemukan masalah, atau menyampaikan pengalaman buruk.

Contoh:
Ulasan: "Barang sesuai deskripsi dan pengiriman cepat."
Label: positif

Ulasan: "Produk rusak dan penjual tidak merespons."
Label: negatif

Balas tepat dengan satu kata: positif atau negatif.
Ulasan yang harus diklasifikasikan: "{review}"
"""


def build_gemini_batch_prompt(reviews: List[str]) -> str:
    """Build one JSON-output prompt for the complete test set."""

    review_lines = "\n".join(
        f"{index}: {review}" for index, review in enumerate(reviews)
    )
    return f"""Anda adalah classifier sentiment untuk ulasan pelanggan e-commerce.

Untuk setiap ulasan, tentukan satu label:
- positif: pelanggan puas atau menyampaikan pengalaman baik.
- negatif: pelanggan kecewa, menemukan masalah, atau menyampaikan pengalaman buruk.

Contoh:
Ulasan: "Barang sesuai deskripsi dan pengiriman cepat."
Label: positif

Ulasan: "Produk rusak dan penjual tidak merespons."
Label: negatif

Kembalikan JSON object yang valid saja, tanpa markdown dan tanpa penjelasan.
Key JSON harus berupa nomor ulasan sebagai string dan value harus tepat "positif" atau "negatif".
Pastikan semua nomor dikembalikan. Contoh format: {{"0": "positif", "1": "negatif"}}

Ulasan yang harus diklasifikasikan:
{review_lines}
"""


def parse_gemini_label(response_text: str) -> Optional[str]:
    """Extract a valid label while tolerating minor formatting differences."""

    normalized = response_text.strip().lower()
    first_word = re.match(r"^(positif|negatif)\b", normalized)
    if first_word:
        return first_word.group(1)

    # Gemini can occasionally stop after a clear prefix when the token limit
    # is too tight, for example returning "posit" or "neg".
    if normalized.startswith("posit"):
        return "positif"
    if normalized.startswith("neg"):
        return "negatif"

    labels_found = re.findall(r"\b(positif|negatif)\b", normalized)
    if len(labels_found) == 1:
        return labels_found[0]
    return None


def predict_with_gemini(
    reviews: pd.Series,
    api_key: str,
    model_name: str = "gemini-3.5-flash-lite",
    temperature: float = 0.0,
) -> Tuple[List[Optional[str]], List[str]]:
    """Classify the complete test set through one structured Gemini request.

    Gemini 3.x uses a thinking level instead of the older temperature-based
    configuration. The low thinking level is enough for this simple
    classification task and keeps latency/cost lower. A single batch request
    also avoids unnecessary free-tier rate-limit errors.
    """

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    review_list = [str(review) for review in reviews]
    predictions: List[Optional[str]] = [None] * len(review_list)
    raw_responses: List[str] = [""] * len(review_list)

    try:
        config_kwargs = {
            "max_output_tokens": max(512, len(review_list) * 6),
            "response_mime_type": "application/json",
        }
        if model_name.startswith("gemini-3"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level="low"
            )
        else:
            # Older models use temperature to control randomness.
            config_kwargs["temperature"] = temperature

        response = client.models.generate_content(
            model=model_name,
            contents=build_gemini_batch_prompt(review_list),
            config=types.GenerateContentConfig(**config_kwargs),
        )
        parsed_response = json.loads((response.text or "").strip())
        if not isinstance(parsed_response, dict):
            raise ValueError("Response Gemini bukan JSON object.")

        for index in range(len(review_list)):
            label_text = str(parsed_response.get(str(index), ""))
            predictions[index] = parse_gemini_label(label_text)
            raw_responses[index] = label_text

        print(f"Gemini batch: {len(review_list)}/{len(review_list)} selesai")
    except Exception as error:
        error_text = f"ERROR: {type(error).__name__}: {error}"
        raw_responses = [error_text] * len(review_list)
        print(f"Gemini batch gagal ({type(error).__name__})")

    return predictions, raw_responses


def calculate_metrics(
    y_true: pd.Series,
    y_pred: List[Optional[str]],
    labels: List[str] = LABELS,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate the requested metrics and confusion matrix."""

    valid_rows = [prediction in labels for prediction in y_pred]
    if not all(valid_rows):
        skipped = len(valid_rows) - sum(valid_rows)
        print(f"Peringatan: {skipped} prediksi tidak valid dilewati dari evaluasi.")

    filtered_true = [truth for truth, valid in zip(y_true, valid_rows) if valid]
    filtered_pred = [prediction for prediction, valid in zip(y_pred, valid_rows) if valid]
    if not filtered_pred:
        raise ValueError("Tidak ada prediksi valid untuk dievaluasi.")

    metrics = {
        "accuracy": accuracy_score(filtered_true, filtered_pred),
        "precision": precision_score(
            filtered_true, filtered_pred, labels=labels, average="binary", pos_label="positif", zero_division=0
        ),
        "recall": recall_score(
            filtered_true, filtered_pred, labels=labels, average="binary", pos_label="positif", zero_division=0
        ),
        "f1_score": f1_score(
            filtered_true, filtered_pred, labels=labels, average="binary", pos_label="positif", zero_division=0
        ),
    }

    matrix = confusion_matrix(filtered_true, filtered_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    report_df = pd.DataFrame(
        classification_report(
            filtered_true,
            filtered_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T
    return metrics, matrix_df, report_df


def save_confusion_matrices(
    matrices: Dict[str, pd.DataFrame],
    output_path: Path = DOCUMENTATION_DIR / "model_comparison_summary.png",
) -> None:
    """Save confusion matrices in one image for the report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    if len(matrices) == 1:
        axes = [axes]

    for axis, (model_name, matrix_df) in zip(axes, matrices.items()):
        sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(f"Confusion Matrix - {model_name}")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_experiment(
    run_gemini: str = "auto",
    data_path: Path = DATA_PATH,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, object]:
    """Run both experiments when credentials are available and save outputs."""

    data = load_dataset(data_path)
    train_data, test_data = split_dataset(data)

    print(f"Jumlah data: {len(data)}")
    print(f"Training set: {len(train_data)} baris")
    print(f"Test set: {len(test_data)} baris")
    print(f"Distribusi label:\n{data['sentiment'].value_counts().to_string()}")

    classic_model = train_classic_model(train_data)
    classic_predictions = classic_model.predict(test_data["review_text"]).tolist()
    classic_metrics, classic_matrix, classic_report = calculate_metrics(
        test_data["sentiment"], classic_predictions
    )

    api_key = os.getenv("GEMINI_API_KEY")
    run_gemini = run_gemini.lower()
    should_run_gemini = run_gemini == "true" or (run_gemini == "auto" and bool(api_key))
    gemini_predictions: List[Optional[str]] = [None] * len(test_data)
    gemini_raw_responses: List[str] = [""] * len(test_data)
    gemini_metrics = None
    gemini_matrix = None
    gemini_report = None

    if should_run_gemini and api_key:
        print("\nMenjalankan inference Gemini pada test set yang sama...")
        gemini_predictions, gemini_raw_responses = predict_with_gemini(
            test_data["review_text"],
            api_key=api_key,
            model_name=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        )
        if any(prediction is not None for prediction in gemini_predictions):
            gemini_metrics, gemini_matrix, gemini_report = calculate_metrics(
                test_data["sentiment"], gemini_predictions
            )
    else:
        print(
            "\nGemini tidak dijalankan. Set GEMINI_API_KEY dan RUN_GEMINI=true/auto "
            "untuk menjalankan inference API."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = test_data[["review_id", "product_name", "review_text", "sentiment"]].copy()
    predictions_df["classic_prediction"] = classic_predictions
    predictions_df["gemini_prediction"] = gemini_predictions
    predictions_df["gemini_raw_response"] = gemini_raw_responses
    predictions_df.to_csv(results_dir / "predictions.csv", index=False)

    summary_rows = [{"model": "Classic ML", **classic_metrics}]
    matrices = {"Classic ML": classic_matrix}
    reports = {"Classic ML": classic_report}
    if gemini_metrics is not None and gemini_matrix is not None and gemini_report is not None:
        summary_rows.append({"model": "Gemini API", **gemini_metrics})
        matrices["Gemini API"] = gemini_matrix
        reports["Gemini API"] = gemini_report

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "model_comparison.csv", index=False)
    save_confusion_matrices(matrices)

    return {
        "data": data,
        "train_data": train_data,
        "test_data": test_data,
        "classic_model": classic_model,
        "classic_predictions": classic_predictions,
        "gemini_predictions": gemini_predictions,
        "gemini_raw_responses": gemini_raw_responses,
        "metrics": summary_df,
        "reports": reports,
        "matrices": matrices,
    }


if __name__ == "__main__":
    experiment_results = run_experiment(run_gemini=os.getenv("RUN_GEMINI", "auto"))
    print("\nRingkasan hasil:")
    print(experiment_results["metrics"].round(4).to_string(index=False))
