import re
import nltk
from nltk.corpus import stopwords
import joblib

# Load model and TF-IDF vectorizer
model = joblib.load("email_phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Email cleaning function (same as training)
def clean_email(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Function to predict
def predict_email(email_text):
    cleaned = clean_email(email_text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return "Phishing 🚨" if pred == 1 else "Legitimate ✅"

# Example usage
if __name__ == "__main__":
    print("=== Email Phishing Detector ===")
    while True:
        email = input("\nEnter email text (or 'exit' to quit):\n")
        if email.lower() == "exit":
            break
        result = predict_email(email)
        print("\nPrediction:", result)