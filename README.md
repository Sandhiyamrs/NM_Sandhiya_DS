# NM_Sandhiya_DS
"Decoding emotions through sentiment analysis of social media conversations"

# 📊 Emotion Detection from Social Media Posts

## 🔍 Overview

This project analyzes social media text (tweets/posts) and identifies emotions such as **joy, anger, sadness, fear, surprise, and neutral** using **NLP and Machine Learning/Deep Learning models**. The goal is to extract emotional insights from large amounts of unstructured social media data.

---

## 🎯 Objectives

* Collect and clean real-world social media data
* Build ML and deep learning models for emotion classification
* Compare different model performances
* Visualize emotional trends
* Deploy a simple user interface for real-time predictions

---

## 🛠 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, nltk, spaCy, scikit-learn, transformers
* **Deployment:** Gradio / Streamlit
* **Tools:** Jupyter Notebook, Google Colab, GitHub

---

## 📂 Dataset

* Source: Kaggle / Public APIs
* Type: Unstructured social media text
* Size: 20K–100K posts
* Columns:

  * `text`
  * `emotion`
  * `date` *(optional)*
* Target: Emotion label

---

## ⚙️ Data Preprocessing

* Removed URLs, emojis, mentions, and symbols
* Lowercasing and tokenization
* Stopword removal
* Lemmatization
* Label encoding
* Train–test split

---

## 📈 EDA Insights

* Joy and neutral posts were most common
* Angry and sad posts often had more uppercase letters and exclamation marks
* Word clouds and frequency charts were created

---

## 🤖 Models Used

* Naive Bayes
* Logistic Regression
* SVM
* Random Forest
* LSTM / Bi-LSTM
* BERT

**Evaluation:** Accuracy, precision, recall, F1 score, confusion matrix

---
