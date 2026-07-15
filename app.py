"""
Spam Email Detector — Streamlit App
Run locally:  streamlit run app.py
"""

import re
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score
)

# ── Import core logic from spam_detector.py ──────────────────────
from spam_detector import (
    preprocess, extract_hand_features, build_features,
    SPAM_WORDS, MODELS
)

MODEL_PATH = "spam_model.pkl"
DATA_PATH  = "spam_data.csv"

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FB; }
    .spam-badge {
        background: #FDECEA; color: #C0392B; font-weight: 700;
        padding: 6px 16px; border-radius: 20px; font-size: 1.1rem;
        border: 1.5px solid #E74C3C;
    }
    .ham-badge {
        background: #EAF6ED; color: #1E8449; font-weight: 700;
        padding: 6px 16px; border-radius: 20px; font-size: 1.1rem;
        border: 1.5px solid #27AE60;
    }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 18px 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card h2 { margin: 4px 0; font-size: 2rem; color: #2C3E50; }
    .metric-card p  { margin: 0; color: #7F8C8D; font-size: 0.85rem; }
    .stTextArea textarea { font-size: 1rem; }
    section[data-testid="stSidebar"] { background-color: #1B2631; }
    section[data-testid="stSidebar"] * { color: #ECF0F1 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load / Train Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model…")
def get_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle["clf"], bundle["tfidf"]

    df = pd.read_csv(DATA_PATH)
    df["label_num"] = (df["label"] == "spam").astype(int)
    X_train, _, y_train, _ = train_test_split(
        df["text"], df["label_num"], test_size=0.2,
        random_state=42, stratify=df["label_num"]
    )
    X_tr_f, _, tfidf = build_features(X_train.tolist(), X_train.tolist())
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_tr_f, y_train)
    return clf, tfidf


@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    df = pd.read_csv(DATA_PATH)
    df["label_num"]  = (df["label"] == "spam").astype(int)
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["upper_ratio"]= df["text"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    df["exclaim"]    = df["text"].str.count("!")
    df["spam_kw"]    = df["text"].apply(
        lambda t: sum(1 for w in t.lower().split() if w in SPAM_WORDS)
    )
    return df


@st.cache_data(show_spinner="Running evaluation…")
def get_eval_results():
    df = get_data()
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        df["text"], df["label_num"], test_size=0.2,
        random_state=42, stratify=df["label_num"]
    )
    X_train_f, X_test_f, tfidf = build_features(X_tr_raw.tolist(), X_te_raw.tolist())

    results = {}
    for name, clf in MODELS.items():
        clf.fit(X_train_f, y_train)
        y_pred = clf.predict(X_test_f)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test_f)[:, 1]
        else:
            y_score = clf.decision_function(X_test_f)
        results[name] = {
            "clf": clf, "y_pred": y_pred, "y_score": y_score,
            "acc": accuracy_score(y_test, y_pred),
            "f1":  f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_score),
            "cm":  confusion_matrix(y_test, y_pred),
        }
    return results, y_test, tfidf


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────
def predict(text, clf, tfidf):
    clean  = preprocess(text)
    tfidf_v = tfidf.transform([clean])
    hf      = csr_matrix(extract_hand_features([text]))
    X       = hstack([tfidf_v, hf])
    label   = clf.predict(X)[0]
    if hasattr(clf, "predict_proba"):
        prob_spam = clf.predict_proba(X)[0][1]
    else:
        raw = clf.decision_function(X)[0]
        prob_spam = 1 / (1 + np.exp(-raw))
    return int(label), float(prob_spam)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Spam Detector")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Predict", "📊 Data Explorer", "🧠 Model Performance", "ℹ️ How It Works"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    df = get_data()
    st.markdown(f"- {len(df):,} messages")
    st.markdown(f"- {df['label_num'].mean():.0%} spam")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("- Naive Bayes\n- Logistic Regression\n- Linear SVM")
    st.markdown("---")
    st.caption("Built with scikit-learn + Streamlit")


clf, tfidf = get_model()


# ═══════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════
if page == "🔍 Predict":
    st.title("🛡️ Spam Email Detector")
    st.markdown("Paste any email or SMS message below to instantly classify it.")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        user_input = st.text_area(
            "✉️ Enter your message",
            height=180,
            placeholder="e.g. WINNER! You've been selected for a FREE prize. Click now!",
        )

        examples = {
            "📧 Legit email": "Hi, just checking in on the project timeline. Can we sync up tomorrow?",
            "🚨 Spam example": "URGENT: Your account will be suspended! Click here to verify NOW and win $1000!",
            "🏦 Phishing": "Your PayPal account is limited. Log in immediately to restore full access.",
            "💬 Normal SMS": "Hey, are you coming to dinner tonight? Mom made pasta.",
        }

        st.markdown("**Try an example:**")
        ecols = st.columns(len(examples))
        for i, (label, text) in enumerate(examples.items()):
            if ecols[i].button(label, use_container_width=True):
                user_input = text
                st.session_state["last_example"] = text

        if "last_example" in st.session_state and not user_input:
            user_input = st.session_state["last_example"]

    with col2:
        if user_input.strip():
            label_num, prob_spam = predict(user_input, clf, tfidf)
            prob_ham = 1 - prob_spam

            st.markdown("### Result")
            if label_num == 1:
                st.markdown('<span class="spam-badge">🚨 SPAM</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="ham-badge">✅ Legitimate</span>', unsafe_allow_html=True)

            st.markdown("#### Confidence")
            st.progress(prob_spam if label_num == 1 else prob_ham)
            st.caption(f"{'Spam' if label_num==1 else 'Ham'} probability: **{(prob_spam if label_num==1 else prob_ham):.1%}**")

            # Gauge bar chart
            fig, ax = plt.subplots(figsize=(4, 1.6))
            bars = ax.barh(["Ham", "Spam"], [prob_ham, prob_spam],
                           color=["#2ECC71", "#E74C3C"], height=0.5)
            for bar, v in zip(bars, [prob_ham, prob_spam]):
                ax.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                        f"{v:.1%}", va="center", fontsize=10, fontweight="bold")
            ax.set_xlim(0, 1.1)
            ax.set_xlabel("Probability")
            ax.spines[["top","right","left"]].set_visible(False)
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Signal breakdown
            hf = extract_hand_features([user_input])[0]
            st.markdown("#### Signal Breakdown")
            sig_cols = st.columns(3)
            sig_cols[0].metric("Characters", int(hf[0]))
            sig_cols[1].metric("UPPER ratio", f"{hf[2]:.0%}")
            sig_cols[2].metric("Spam keywords", int(hf[6]))
            sig_cols2 = st.columns(3)
            sig_cols2[0].metric("Words", int(hf[1]))
            sig_cols2[1].metric("Exclamations", int(hf[4]))
            sig_cols2[2].metric("URLs found", int(hf[5]))
        else:
            st.info("👈 Enter a message to get started.")

    # Batch prediction
    st.markdown("---")
    st.subheader("📋 Batch Prediction (CSV)")
    uploaded = st.file_uploader("Upload a CSV with a `text` column", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if "text" in batch_df.columns:
            preds = [predict(t, clf, tfidf) for t in batch_df["text"]]
            batch_df["prediction"] = ["spam" if p[0] else "ham" for p in preds]
            batch_df["prob_spam"]  = [round(p[1], 4) for p in preds]
            st.dataframe(batch_df, use_container_width=True)
            st.download_button("⬇️ Download results", batch_df.to_csv(index=False),
                               "spam_results.csv", "text/csv")
        else:
            st.error("CSV must have a `text` column.")


# ═══════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ═══════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    for col, title, val in [
        (c1, "Total Messages", f"{len(df):,}"),
        (c2, "Ham (Legit)", f"{(df['label']=='ham').sum():,}"),
        (c3, "Spam", f"{(df['label']=='spam').sum():,}"),
        (c4, "Spam Rate", f"{df['label_num'].mean():.1%}"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <p>{title}</p>
            <h2>{val}</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["label"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=counts.index, autopct="%1.1f%%",
            colors=["#4C9BE8", "#E85B4C"], startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        for t in autotexts: t.set_fontweight("bold")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.subheader("Message Length by Class")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color in [("ham","#4C9BE8"), ("spam","#E85B4C")]:
            ax.hist(df[df["label"]==label]["char_count"], bins=35,
                    alpha=0.6, label=label, color=color)
        ax.set_xlabel("Character Count"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Uppercase Ratio")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color in [("ham","#4C9BE8"), ("spam","#E85B4C")]:
            ax.hist(df[df["label"]==label]["upper_ratio"], bins=30,
                    alpha=0.6, label=label, color=color)
        ax.set_xlabel("Fraction of uppercase chars"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right2:
        st.subheader("Feature Comparison")
        feat_df = df.groupby("label")[["word_count","upper_ratio","exclaim","spam_kw"]].mean().T
        feat_df.columns.name = None
        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = range(len(feat_df))
        w = 0.35
        ax.bar([i - w/2 for i in x], feat_df["ham"],  width=w, label="ham",  color="#4C9BE8")
        ax.bar([i + w/2 for i in x], feat_df["spam"], width=w, label="spam", color="#E85B4C")
        ax.set_xticks(list(x))
        ax.set_xticklabels(feat_df.index, rotation=15, ha="right")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.subheader("Sample Messages")
    filter_label = st.selectbox("Filter by", ["All", "ham", "spam"])
    sample_df = df if filter_label == "All" else df[df["label"] == filter_label]
    st.dataframe(
        sample_df[["label","text","char_count","word_count","exclaim"]].sample(
            min(20, len(sample_df)), random_state=1
        ).reset_index(drop=True),
        use_container_width=True
    )


# ═══════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "🧠 Model Performance":
    st.title("🧠 Model Performance")

    with st.spinner("Training & evaluating all models…"):
        results, y_test, eval_tfidf = get_eval_results()

    # Score cards
    st.subheader("Comparison")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        col.markdown(f"""
        <div class="metric-card">
            <p style='font-size:0.9rem;font-weight:600'>{name}</p>
            <h2>{res['f1']:.3f}</h2>
            <p>F1 Score</p>
            <p>Acc: {res['acc']:.3f} · AUC: {res['auc']:.3f}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    selected_model = st.selectbox("Select model for detailed view", list(results.keys()))
    res = results[selected_model]

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham","spam"], yticklabels=["ham","spam"],
                    ax=ax, linewidths=0.5, cbar=False, annot_kws={"size":14})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, res["y_score"])
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(fpr, tpr, color="#4C9BE8", lw=2.5, label=f"AUC = {res['auc']:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#4C9BE8")
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right"); ax.spines[["top","right"]].set_visible(False)
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Feature importance for LR
    if selected_model == "Logistic Regression":
        st.subheader("Top Predictive Features")
        feature_names = eval_tfidf.get_feature_names_out()
        n_tfidf = len(feature_names)
        coef = res["clf"].coef_[0][:n_tfidf]
        n_show = st.slider("Features to show", 10, 30, 20)

        top_spam = np.argsort(coef)[-n_show:][::-1]
        top_ham  = np.argsort(coef)[:n_show]

        fc1, fc2 = st.columns(2)
        for col, idxs, title, color in [
            (fc1, top_spam, "🚨 Spam Signals", "#E85B4C"),
            (fc2, top_ham,  "✅ Ham Signals",  "#4C9BE8"),
        ]:
            with col:
                fig, ax = plt.subplots(figsize=(5, n_show * 0.28 + 1))
                feats = [feature_names[i] for i in idxs]
                vals  = [abs(coef[i]) for i in idxs]
                ax.barh(feats[::-1], vals[::-1], color=color, edgecolor="white")
                ax.set_title(title, fontweight="bold")
                ax.set_xlabel("|Coefficient|")
                ax.spines[["top","right"]].set_visible(False)
                fig.patch.set_alpha(0)
                st.pyplot(fig, use_container_width=True)
                plt.close()


# ═══════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ═══════════════════════════════════════════════
elif page == "ℹ️ How It Works":
    st.title("ℹ️ How It Works")

    st.markdown("""
    ### Pipeline Overview

    This app runs a classic **NLP + ML** pipeline with a hybrid feature approach:

    ---

    #### 1. 🧹 Text Preprocessing
    Each message goes through:
    - **Lowercasing** — normalise casing
    - **URL tokenisation** — replace links with `urltoken`
    - **Number tokenisation** — replace digits with `numtoken`
    - **Punctuation removal** — strip non-alphabetic characters
    - **Stopword filtering** — remove common English words *except* spam-signal words

    ---

    #### 2. 🔢 Feature Engineering

    Two feature sets are **combined** using `scipy.sparse.hstack`:

    | Type | Description |
    |---|---|
    | **TF-IDF** (1–2 grams) | 10,000 weighted text features |
    | **char_count** | Total message length |
    | **word_count** | Number of words |
    | **upper_ratio** | Fraction of uppercase characters |
    | **digit_count** | Number of digits |
    | **exclaim_count** | Number of `!` marks |
    | **url_count** | Number of URLs/links |
    | **spam_keyword_count** | Known spam vocabulary hits |

    ---

    #### 3. 🤖 Models Compared

    | Model | Strengths |
    |---|---|
    | **Naive Bayes** | Fast, excellent baseline for text |
    | **Logistic Regression** | Interpretable, strong with TF-IDF |
    | **Linear SVM** | Best margin-based decision boundary |

    The **best model by F1 score** is saved to `spam_model.pkl` and used for predictions.

    ---

    #### 4. 📈 Why F1, not Accuracy?

    Spam datasets are **imbalanced** (typically ~20–30% spam).  
    A dumb model that classifies everything as *ham* would get 80%+ accuracy — but F1 penalises  
    both false positives (legit emails marked spam) and false negatives (spam that slips through).

    ---

    #### 📦 Tech Stack
    `scikit-learn` · `pandas` · `numpy` · `scipy` · `matplotlib` · `seaborn` · `streamlit`
    """)
