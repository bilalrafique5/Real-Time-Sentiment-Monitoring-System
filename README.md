# Real-Time Sentiment Monitoring System

A **Real-Time Sentiment Monitoring System** that analyzes live tweets using the **X (Twitter) API** and allows users to choose between **VADER** and **DistilBERT** for sentiment analysis. The application provides real-time insights through an interactive **Streamlit** interface and uses **Firebase** as the backend for data storage and caching.

---

## 🚀 Features

* 🔴 **Real-time tweet fetching** using the X API (via Tweepy)
* 🧠 **Dual sentiment analysis models**:

  * **VADER** (rule-based, fast)
  * **DistilBERT** (transformer-based, deep learning)
* 🎛️ **Streamlit-based interactive UI**
* ☁️ **Firebase backend** to store:

  * Search queries
  * Usernames
  * Tweets
  * Sentiment scores from both models
  * Timestamps
* ⚡ **Smart caching mechanism**:

  * If a user searches the same query again within **10 hours**, results are fetched from Firebase instead of calling the X API again
  * Reduces API usage and improves response time
* 📊 **Visual analytics** using Plotly and WordCloud

---

## 🏗️ System Architecture

```
User (Streamlit UI)
        ↓
X API (Tweepy)  ←── Cache Check (Firebase, 10 hours)
        ↓
Sentiment Analysis (VADER / DistilBERT)
        ↓
Firebase Database
        ↓
Visualization (Plots & WordClouds)
```

---

## 📦 Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python + Firebase
* **Database**: Firebase (Firestore / Realtime DB)
* **APIs**: X (Twitter) API
* **NLP Models**: VADER, DistilBERT

---

## 📋 Requirements

Install the required Python libraries using:

```bash
pip install tweepy streamlit pandas numpy nltk vaderSentiment transformers torch plotly wordcloud firebase-admin python-dateutil
```

### Dependencies List

* tweepy
* streamlit
* pandas
* numpy
* nltk
* vaderSentiment
* transformers
* torch
* plotly
* wordcloud
* firebase-admin
* python-dateutil

---

## 🔑 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/real-time-sentiment-monitoring.git
cd real-time-sentiment-monitoring
```

### 2️⃣ X API Configuration

* Create a developer account on **X (Twitter)**
* Generate API keys and access tokens
* Add them to your environment variables or configuration file

### 3️⃣ Firebase Configuration

* Create a Firebase project
* Enable Firestore or Realtime Database
* Generate a **service account key (JSON)**
* Initialize Firebase in your project using `firebase-admin`

```python
import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 📊 Output & Visualizations

* Sentiment distribution (Positive / Neutral / Negative)
* Comparative analysis between VADER & DistilBERT
* Interactive charts using Plotly
* WordCloud of most frequent terms in tweets

---

## 🧠 Sentiment Models

### VADER

* Rule-based sentiment analyzer
* Fast and lightweight
* Ideal for social media text

### DistilBERT

* Transformer-based deep learning model
* More accurate contextual understanding
* Computationally heavier

Users can select the model directly from the UI.

---

## 🧪 Caching Logic (10 Hours)

* Every query is stored with a timestamp in Firebase
* When the same query is searched again:

  * If the time difference ≤ **10 hours**, cached data is returned
  * Otherwise, fresh tweets are fetched from X API

---

## 📌 Future Enhancements

* Live sentiment streaming dashboard
* Support for multilingual sentiment analysis
* User authentication with Firebase Auth
* Topic-wise sentiment comparison
* Deployment on cloud platforms

---

## 👨‍💻 Author

**Bilal Rafique**
BS (Computer Science)

---

## 📜 License

This project is for educational and research purposes.

---

⭐ If you like this project, consider giving it a star on GitHub!
