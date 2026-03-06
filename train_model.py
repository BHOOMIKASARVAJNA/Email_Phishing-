import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------
# Download stopwords (one-time)
# -----------------------------
nltk.download('stopwords')

# -----------------------------
# STEP 1: Load the dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\bhoom\OneDrive\Desktop\email_phish\email\phishing_email.csv")

# Rename columns if needed
data = data.rename(columns={"text_combined": "email"})

# Separate features and labels
emails = data["email"]
labels = data["label"]

# -----------------------------
# STEP 2: Text Cleaning Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_email(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'\d+', '', text)              # remove numbers
    text = re.sub(r'[^\w\s]', '', text)          # remove punctuation
    words = [word for word in text.split() if word not in stop_words]  # remove stopwords
    return " ".join(words)

# Apply cleaning
print("Cleaning email texts...")
cleaned_emails = emails.apply(clean_email)

# -----------------------------
# STEP 3: Convert text to TF-IDF features
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(cleaned_emails)
y = labels

# -----------------------------
# STEP 4: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset loaded and cleaned successfully.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# -----------------------------
# STEP 5: Train Naive Bayes Model
# -----------------------------
print("\nTraining Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model trained successfully!")

# -----------------------------
# STEP 6: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {accuracy*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# STEP 7: Save Model and Vectorizer
# -----------------------------
joblib.dump(model, "email_phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")