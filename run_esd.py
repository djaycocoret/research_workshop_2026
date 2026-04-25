import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
SAMPLE_RATE = 16000
RANDOM_STATE = 42
MAX_ITER = 2000

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_EMBEDDINGS_PATH = OUTPUTS_DIR / "esd_embeddings_all.pkl"
RESULTS_PATH = OUTPUTS_DIR / "experiment_results.csv"

DEVICE = torch.device("cpu")


def load_wav2vec_model(model_name: str = MODEL_NAME):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return feature_extractor, model


def get_language_from_speaker(speaker_id: str) -> str:
    speaker_num = int(speaker_id)
    if 1 <= speaker_num <= 10:
        return "mandarin"
    if 11 <= speaker_num <= 20:
        return "english"
    raise ValueError(f"Unknown speaker id: {speaker_id}")


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr


def extract_embedding(
    file_path: str,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    sample_rate: int = SAMPLE_RATE,
):
    audio, _ = load_audio(file_path, sample_rate=sample_rate)
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    return embedding


def process_single_speaker(
    speaker_dir: Path,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
):
    speaker_id = speaker_dir.name
    language = get_language_from_speaker(speaker_id)
    data = []

    for emotion_dir in sorted(speaker_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name

        for file_path in sorted(emotion_dir.glob("*.wav")):
            try:
                embedding = extract_embedding(
                    str(file_path),
                    feature_extractor=feature_extractor,
                    model=model,
                )
            except Exception as e:
                print(f"Skipping file: {file_path}")
                print(f"Error: {e}")
                continue

            data.append(
                {
                    "file": file_path.name,
                    "speaker": speaker_id,
                    "language": language,
                    "emotion": emotion,
                    "embedding": embedding,
                }
            )

    return pd.DataFrame(data)


def process_and_save_per_speaker(
    root_dir: Path,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    outputs_dir: Path,
):
    for speaker_dir in sorted(root_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        if not speaker_id.isdigit():
            continue

        speaker_output_path = outputs_dir / f"embeddings_{speaker_id}.pkl"

        if speaker_output_path.exists():
            print(f"Skipping speaker {speaker_id}, already saved")
            continue

        print(f"Processing speaker {speaker_id}...")
        speaker_df = process_single_speaker(
            speaker_dir=speaker_dir,
            feature_extractor=feature_extractor,
            model=model,
        )

        speaker_df.to_pickle(speaker_output_path)
        print(f"Saved speaker embeddings to: {speaker_output_path}")


def combine_speaker_embeddings(outputs_dir: Path, combined_path: Path):
    dfs = []

    for speaker_file in sorted(outputs_dir.glob("embeddings_????.pkl")):
        dfs.append(pd.read_pickle(speaker_file))

    if not dfs:
        raise FileNotFoundError("No per-speaker embedding files were found.")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_pickle(combined_path)

    return full_df


def load_embeddings(input_path: Path):
    return pd.read_pickle(input_path)


def prepare_features_and_labels(df: pd.DataFrame):
    X = np.stack(df["embedding"].values)
    y = df["emotion"].values
    return X, y


def train_logistic_regression(X_train, y_train, max_iter: int = MAX_ITER):
    clf = LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }


def run_within_language_experiment(df: pd.DataFrame, language: str):
    subset = df[df["language"] == language].copy()

    X, y = prepare_features_and_labels(subset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    clf = train_logistic_regression(X_train, y_train)
    results = evaluate_model(clf, X_test, y_test)

    return {
        "experiment": f"{language}_to_{language}",
        "train_language": language,
        "test_language": language,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": results["accuracy"],
        "f1_weighted": results["f1_weighted"],
    }


def run_cross_language_experiment(df: pd.DataFrame, train_language: str, test_language: str):
    train_df = df[df["language"] == train_language].copy()
    test_df = df[df["language"] == test_language].copy()

    X_train, y_train = prepare_features_and_labels(train_df)
    X_test, y_test = prepare_features_and_labels(test_df)

    clf = train_logistic_regression(X_train, y_train)
    results = evaluate_model(clf, X_test, y_test)

    return {
        "experiment": f"{train_language}_to_{test_language}",
        "train_language": train_language,
        "test_language": test_language,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": results["accuracy"],
        "f1_weighted": results["f1_weighted"],
    }


def run_all_experiments(df: pd.DataFrame):
    results = []

    results.append(run_within_language_experiment(df, "mandarin"))
    results.append(run_within_language_experiment(df, "english"))
    results.append(run_cross_language_experiment(df, "mandarin", "english"))
    results.append(run_cross_language_experiment(df, "english", "mandarin"))

    return pd.DataFrame(results)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Outputs folder: {OUTPUTS_DIR}")

    if COMBINED_EMBEDDINGS_PATH.exists():
        print("Loading combined embeddings...")
        df = load_embeddings(COMBINED_EMBEDDINGS_PATH)
    else:
        print("Loading model...")
        feature_extractor, model = load_wav2vec_model()

        print("Processing speakers and saving per speaker...")
        process_and_save_per_speaker(
            root_dir=DATA_DIR,
            feature_extractor=feature_extractor,
            model=model,
            outputs_dir=OUTPUTS_DIR,
        )

        print("Combining all saved speaker embeddings...")
        df = combine_speaker_embeddings(
            outputs_dir=OUTPUTS_DIR,
            combined_path=COMBINED_EMBEDDINGS_PATH,
        )

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset summary:")
    print(f"Number of files: {len(df)}")
    print(f"Embedding shape: {df.iloc[0]['embedding'].shape}")
    print("\nFiles per language:")
    print(df["language"].value_counts())
    print("\nFiles per emotion:")
    print(df["emotion"].value_counts())

    print("\nRunning experiments...")
    results_df = run_all_experiments(df)

    print("\nResults:")
    print(results_df[["experiment", "n_train", "n_test", "accuracy", "f1_weighted"]])

    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved results to: {RESULTS_PATH}")
    print(f"Results file exists: {RESULTS_PATH.exists()}")
    print(f"Combined embeddings file exists: {COMBINED_EMBEDDINGS_PATH.exists()}")


if __name__ == "__main__":
    main()