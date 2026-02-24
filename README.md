
# 🎬 IMDB Sentiment Analysis (High Accuracy NLP Project)

## 📌 Project Overview

This project implements a complete end-to-end Sentiment Analysis system that classifies IMDB movie reviews as **positive** or **negative** using classical Natural Language Processing (NLP) techniques. The goal of this repository is to demonstrate a clean, production-style machine learning workflow that is simple, fast, and highly effective on real-world text data.

Sentiment analysis is widely used in industry for understanding customer feedback, product reviews, and public opinion. In this implementation, we use the popular **IMDB Dataset of 50K Movie Reviews**, which provides a balanced and well-structured benchmark for binary sentiment classification.

The pipeline converts raw text into numerical features using **TF-IDF vectorization** with unigram and bigram support to capture contextual meaning. A **Linear Support Vector Classifier (LinearSVC)** is then trained on these features, as it performs extremely well on high-dimensional sparse text data while remaining computationally efficient.

The model typically achieves **88–94% accuracy**, making it a strong baseline NLP project suitable for academic submissions, internships, and beginner ML portfolios.

---

## 🎯 Key Features

- End-to-end NLP pipeline  
- Clean and modular project structure  
- High accuracy with fast training  
- TF-IDF feature engineering  
- LinearSVC classifier  
- Model serialization using joblib  
- Easy to extend and deploy  

---

## 📊 Dataset

**Name:** IMDB Dataset of 50K Movie Reviews  

Each record contains:

- `review` → movie review text  
- `sentiment` → positive / negative label  

The dataset is balanced, which helps the model learn effectively without heavy preprocessing.

---

## 🧠 Machine Learning Pipeline

The workflow followed in this project:

1. Load and inspect dataset  
2. Text preprocessing (basic cleaning via TF-IDF)  
3. Train-test split (80/20, stratified)  
4. Feature extraction using TF-IDF  
5. Model training using LinearSVC  
6. Performance evaluation  
7. Model saving for reuse  

---

## ⚙️ Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF Vectorizer  
- Linear Support Vector Machine  
- Joblib  

---

## 📁 Project Structure

```

SENTIMENTAL_ANALYSIS/
│
├── data/
│   └── IMDB Dataset.csv
│
├── model/
│   └── model.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md

````

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
````

---

## ▶️ Training the Model

Run the training script:

```bash
python train.py
```

After training completes, the model will be saved to:

```
model/model.pkl
```

---

## 🧪 Example Prediction

```python
import joblib

model = joblib.load("model/model.pkl")

sample_review = ["This movie was absolutely fantastic and emotional"]
prediction = model.predict(sample_review)

print("Prediction:", prediction[0])
```

---

## 📈 Model Performance

Typical results:

* Accuracy: **0.88 – 0.94**
* Strong precision and recall balance
* Fast training time
* Good generalization

This makes the model suitable as a solid baseline sentiment classifier.

---

## 🔮 Future Improvements

Possible extensions:

* Deep learning models (LSTM, BERT)
* Hyperparameter tuning
* Advanced text preprocessing
* Streamlit web deployment
* Model explainability (SHAP/LIME)
* REST API integration

---

## 🎓 Learning Outcomes

Through this project, you demonstrate:

* Practical NLP skills
* Feature engineering for text
* Classical ML model building
* Evaluation and validation
* Clean ML project structuring

These are highly relevant skills for data science and machine learning roles.

---

## 👩‍💻 Author

Built as part of a machine learning and NLP portfolio project.

```
::contentReference[oaicite:0]{index=0}
```
