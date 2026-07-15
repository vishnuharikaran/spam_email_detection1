# 🛡️ Spam Email Detector

An end-to-end spam detection project built with **scikit-learn** and deployed as an interactive **Streamlit** web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🚀 Live Demo

> Replace the badge URL above with your Streamlit Cloud link after deployment.

---

## 📁 Project Structure

```
spam-detector/
├── app.py               # Streamlit web app
├── spam_detector.py     # Core ML pipeline (preprocessing, features, models)
├── spam_data.csv        # Dataset (5,000 labelled messages)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Features

- **Predict** — classify any message as spam or ham with confidence score
- **Batch prediction** — upload a CSV and download results
- **Data Explorer** — interactive EDA charts (class distribution, message length, etc.)
- **Model Performance** — confusion matrix, ROC curves, feature importance
- **How It Works** — full pipeline explainer

---

## 🧠 ML Pipeline

1. **Preprocessing** — lowercase, URL/number tokenisation, stopword removal  
2. **TF-IDF** — unigrams + bigrams, 10,000 features, sublinear TF scaling  
3. **Hand-crafted features** — uppercase ratio, exclamation count, spam keywords, etc.  
4. **Models** — Naive Bayes · Logistic Regression · Linear SVM  
5. **Evaluation** — Accuracy, F1, ROC-AUC, confusion matrix  

---

## 🏃 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub  
2. Go to [share.streamlit.io](https://share.streamlit.io)  
3. Click **New app** → select your repo → set **Main file path** to `app.py`  
4. Click **Deploy** — done in ~2 minutes!

---

## 📊 Dataset

- 5,000 labelled messages (3,900 ham / 1,100 spam)  
- 22% spam rate — realistic class imbalance  
- Swap `spam_data.csv` with the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) for a real-world benchmark  

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `pandas` · `numpy` · `scipy` · `matplotlib` · `seaborn` · `streamlit`
