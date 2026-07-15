"""
=============================================================
  Spam Email Detector — End-to-End Data Science Project
=============================================================
Pipeline:
  1. Data Loading & Exploration (EDA)
  2. Text Preprocessing
  3. Feature Engineering (TF-IDF + hand-crafted features)
  4. Model Training (Naive Bayes, Logistic Regression, SVM)
  5. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
  6. Model Persistence
  7. Interactive Predictor
"""

import re
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score
)
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────
# 1. DATA LOADING & EDA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[["label", "text"]].dropna()
    df["label_num"] = (df["label"] == "spam").astype(int)
    print(f"\n{'='*50}")
    print(f"  Dataset loaded: {len(df):,} rows")
    print(f"{'='*50}")
    print(df["label"].value_counts().to_string())
    print(f"\nClass balance: {df['label_num'].mean():.1%} spam")
    return df


def explore_data(df: pd.DataFrame):
    """Print EDA summary and save plots."""
    df = df.copy()
    df["char_count"]  = df["text"].str.len()
    df["word_count"]  = df["text"].str.split().str.len()
    df["upper_ratio"] = df["text"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    df["digit_count"] = df["text"].str.count(r"\d")
    df["exclaim"]     = df["text"].str.count(r"!")
    df["url_count"]   = df["text"].str.count(r"http|www|click")

    print("\n── EDA: Per-class statistics ──")
    print(df.groupby("label")[["char_count","word_count","upper_ratio","exclaim"]].mean().round(2))

    # ── Plot 1: Class distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Spam Detector — Exploratory Data Analysis", fontsize=15, fontweight="bold")

    colors = {"ham": "#4C9BE8", "spam": "#E85B4C"}

    ax = axes[0, 0]
    counts = df["label"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[colors[l] for l in counts.index], edgecolor="white", width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{v:,}", ha="center", fontweight="bold")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.spines[["top","right"]].set_visible(False)

    # ── Plot 2: Message length distribution
    ax = axes[0, 1]
    for label, grp in df.groupby("label"):
        ax.hist(grp["char_count"], bins=40, alpha=0.6, label=label, color=colors[label])
    ax.set_title("Message Length (characters)")
    ax.set_xlabel("Character Count")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # ── Plot 3: Uppercase ratio
    ax = axes[1, 0]
    for label, grp in df.groupby("label"):
        ax.hist(grp["upper_ratio"], bins=30, alpha=0.6, label=label, color=colors[label])
    ax.set_title("Uppercase Ratio")
    ax.set_xlabel("Ratio of uppercase characters")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # ── Plot 4: Exclamation marks
    ax = axes[1, 1]
    ax.boxplot(
        [df[df["label"]=="ham"]["exclaim"], df[df["label"]=="spam"]["exclaim"]],
        labels=["ham", "spam"],
        patch_artist=True,
        boxprops=dict(facecolor="#DDE9F5"),
        medianprops=dict(color="#333", linewidth=2),
    )
    ax.set_title("Exclamation Marks per Message")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "eda_plots.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  EDA plot saved → {out}")
    return df


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

SPAM_WORDS = {
    "free","winner","won","prize","claim","cash","urgent","offer",
    "guarantee","credit","loan","click","subscribe","buy","discount",
    "casino","lottery","earn","income","profit","reward","selected",
    "congratulations","act","now","limited","exclusive","account",
    "verify","suspend","confirm","password","bank","paypal","bitcoin",
}

def preprocess(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\b\d+\b", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    # Keep spam-signal words; remove common English stopwords
    stopwords = {"i","me","my","we","our","you","your","he","she","it",
                 "they","the","a","an","and","or","but","in","on","at",
                 "to","for","of","with","as","is","are","was","were",
                 "be","been","have","has","had","do","did","will","would",
                 "can","could","not","no","so","if","by","from","that","this"}
    tokens = [t for t in tokens if t not in stopwords or t in SPAM_WORDS]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# 3. HAND-CRAFTED FEATURES
# ─────────────────────────────────────────────

def extract_hand_features(texts):
    """Return a (n_samples, 7) numeric feature matrix."""
    rows = []
    for t in texts:
        char_count  = len(t)
        word_count  = len(t.split())
        upper_ratio = sum(1 for c in t if c.isupper()) / max(char_count, 1)
        digit_count = len(re.findall(r"\d", t))
        exclaim     = t.count("!")
        url_count   = len(re.findall(r"http|www|click", t, re.I))
        spam_kw     = sum(1 for w in t.lower().split() if w in SPAM_WORDS)
        rows.append([char_count, word_count, upper_ratio,
                     digit_count, exclaim, url_count, spam_kw])
    return np.array(rows, dtype=float)


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(X_train_raw, X_test_raw):
    """Combine TF-IDF + hand-crafted numeric features."""
    X_train_clean = [preprocess(t) for t in X_train_raw]
    X_test_clean  = [preprocess(t) for t in X_test_raw]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=10_000,
        sublinear_tf=True,    # dampen high-frequency terms
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf  = tfidf.transform(X_test_clean)

    X_train_hf = csr_matrix(extract_hand_features(X_train_raw))
    X_test_hf  = csr_matrix(extract_hand_features(X_test_raw))

    X_train = hstack([X_train_tfidf, X_train_hf])
    X_test  = hstack([X_test_tfidf,  X_test_hf])

    return X_train, X_test, tfidf


# ─────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

MODELS = {
    "Naive Bayes":          MultinomialNB(alpha=0.1),
    "Logistic Regression":  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    "Linear SVM":           LinearSVC(C=0.5, max_iter=2000),
}


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train all models, print reports, save comparison plot."""
    results = {}

    print(f"\n{'='*50}")
    print("  Model Evaluation")
    print(f"{'='*50}")

    for name, clf in MODELS.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # ROC-AUC (use decision_function / predict_proba)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_score)

        results[name] = {"clf": clf, "y_pred": y_pred,
                         "y_score": y_score, "acc": acc, "f1": f1, "auc": auc}

        print(f"\n── {name} ──")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["ham","spam"]))

    return results


def save_evaluation_plots(results, y_test):
    """Confusion matrices + ROC curves side by side."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 9))
    fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")

    colors = ["#4C9BE8", "#6EC574", "#E8A14C"]

    for i, (name, res) in enumerate(results.items()):
        # Confusion matrix
        ax = axes[0, i]
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham","spam"], yticklabels=["ham","spam"],
                    ax=ax, linewidths=0.5, cbar=False)
        ax.set_title(f"{name}\nConf. Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # ROC curve
        ax = axes[1, i]
        fpr, tpr, _ = roc_curve(y_test, res["y_score"])
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"AUC = {res['auc']:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.08, color=colors[i])
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}\nROC Curve")
        ax.legend(loc="lower right")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "model_evaluation.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Evaluation plot saved → {out}")


def save_feature_importance(tfidf, best_clf):
    """Top TF-IDF features by class weight (Logistic Regression)."""
    if not hasattr(best_clf, "coef_"):
        return
    feature_names = tfidf.get_feature_names_out()
    n_tfidf = len(feature_names)
    # coef_ includes hand-crafted features appended after TF-IDF — slice only TF-IDF part
    coef = best_clf.coef_[0][:n_tfidf]

    top_spam = np.argsort(coef)[-20:][::-1]
    top_ham  = np.argsort(coef)[:20]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Top Predictive Features (Logistic Regression)", fontsize=13, fontweight="bold")

    for ax, idxs, title, color in [
        (axes[0], top_spam, "Top 20 Spam Signals", "#E85B4C"),
        (axes[1], top_ham,  "Top 20 Ham Signals",  "#4C9BE8"),
    ]:
        feats = [feature_names[i] for i in idxs]
        vals  = [abs(coef[i]) for i in idxs]
        ax.barh(feats[::-1], vals[::-1], color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("|Coefficient|")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved → {out}")


# ─────────────────────────────────────────────
# 6. MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_best_model(results, tfidf, path):
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = {"name": best_name, "clf": results[best_name]["clf"], "tfidf": tfidf}
    with open(path, "wb") as f:
        pickle.dump(best, f)
    print(f"\n  Best model: {best_name}  (F1={results[best_name]['f1']:.4f})")
    print(f"  Saved → {path}")
    return best


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# 7. INTERACTIVE PREDICTOR
# ─────────────────────────────────────────────

class SpamDetector:
    """Thin wrapper for inference on new messages."""

    def __init__(self, model_path: str):
        bundle = load_model(model_path)
        self.name  = bundle["name"]
        self.clf   = bundle["clf"]
        self.tfidf = bundle["tfidf"]

    def predict(self, text: str) -> dict:
        clean   = preprocess(text)
        tfidf_v = self.tfidf.transform([clean])
        hf      = csr_matrix(extract_hand_features([text]))
        X       = hstack([tfidf_v, hf])

        label = self.clf.predict(X)[0]
        if hasattr(self.clf, "predict_proba"):
            prob_spam = self.clf.predict_proba(X)[0][1]
        elif hasattr(self.clf, "decision_function"):
            raw = self.clf.decision_function(X)[0]
            prob_spam = 1 / (1 + np.exp(-raw))   # sigmoid
        else:
            prob_spam = float(label)

        return {
            "label":      "spam" if label == 1 else "ham",
            "confidence": round(prob_spam if label == 1 else 1 - prob_spam, 4),
            "prob_spam":  round(prob_spam, 4),
        }

    def batch_predict(self, texts):
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH  = os.path.join(OUTPUT_DIR, "spam_data.csv")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_model.pkl")

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA
    explore_data(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"],
        test_size=0.2, random_state=42, stratify=df["label_num"]
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 4. Features
    X_train_feat, X_test_feat, tfidf = build_features(X_train.tolist(), X_test.tolist())

    # 5. Evaluate
    results = evaluate_models(X_train_feat, X_test_feat, y_train, y_test)
    save_evaluation_plots(results, y_test)
    save_feature_importance(tfidf, results["Logistic Regression"]["clf"])

    # 6. Save
    best = save_best_model(results, tfidf, MODEL_PATH)

    # 7. Demo
    print(f"\n{'='*50}")
    print("  Interactive Demo")
    print(f"{'='*50}")
    detector = SpamDetector(MODEL_PATH)
    demo_messages = [
        "Hey, are we still on for lunch on Friday?",
        "WINNER! You've been selected for a FREE iPhone. Click to claim NOW!",
        "Your invoice #INV-2091 has been processed. Thank you.",
        "URGENT: Your account will be suspended unless you verify now.",
        "Can you review the pull request when you have a moment?",
        "Earn 5000 dollars a week from home! No experience needed!",
    ]
    for msg in demo_messages:
        result = detector.predict(msg)
        tag    = "🚨 SPAM" if result["label"] == "spam" else "✅ HAM "
        print(f"  {tag}  [{result['prob_spam']:.0%} spam]  {msg[:60]}")

    print(f"\n{'='*50}")
    print("  All done! Files created:")
    for f in ["eda_plots.png","model_evaluation.png","feature_importance.png","spam_model.pkl"]:
        print(f"    • {f}")
    print(f"{'='*50}\n")
