# 📧 Spam Email Detector

A simple **Machine Learning-based Spam Email Detector** built with **Python, Scikit-learn, and Streamlit**.  
This app classifies emails as **Ham (Not Spam)** or **Spam** with confidence scores.

---

## 🚀 Features
- Classifies email text as **Ham** or **Spam**
- Provides a **confidence score** (percentage) for predictions
- Built using:
  - **Linear SVC (Support Vector Classifier)**
  - **TF-IDF Vectorizer**
- Deployed as a **Streamlit web app**

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **scikit-learn**
- **pandas**
- **numpy**
- **joblib**
- **Streamlit**

---

## 📂 Project Structure
```
SPAM_email_detection/
│── models/
│   ├── linear_svc_model.pkl
│   ├── tfidf_vectorizer.pkl
│
│── app.py                # Streamlit app
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## ⚡ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/SPAM_email_detection.git
   cd SPAM_email_detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Linux/Mac
   venv\Scripts\activate         # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🎯 Usage
1. Paste an email message in the text box.  
2. Click **"🔍 Predict"**.  
3. See if it’s **Ham (Not Spam)** ✅ or **Spam** 🚨.  
4. Confidence score will also be displayed.  

---

## 📝 Example
```
Input: "Congratulations! You've won a free iPhone. Click here to claim now."
Output: 🚨 This email is Spam (Confidence: 97.3%)
```

---

## 🔮 Future Improvements
- Train with a larger dataset (Enron / SpamAssassin)  
- Add **email subject + sender features**  
- Deploy online with **Streamlit Cloud / HuggingFace Spaces / Heroku**  

---

## 👨‍💻 Author
**Varun Kumar**  
📧 [varunkr6302@gmail.com](mailto:varunkr6302@gmail.com)  
🔗 [GitHub Profile](https://github.com/<your-username>)  
