import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv('IFND.csv', encoding='iso-8859-1')

# Encode labels: 0 = Real, 1 = Fake
data['fake'] = data['Label'].str.upper().map({'TRUE': 0, 'FAKE': 1})


# Check class distribution
print("Class distribution:")
print(data['fake'].value_counts())
print(f"\nClass balance: {data['fake'].value_counts(normalize=True)}")

# Handle missing values by filling with empty strings
data['Statement'] = data['Statement'].fillna('')
data['Web'] = data['Web'].fillna('')
data['Category'] = data['Category'].fillna('')

# Prepare input text by concatenating columns with spaces
X = data['Statement'] + " " + data['Web'] + " " + data['Category']
y = data['fake']

# Split dataset with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training class distribution:\n{y_train.value_counts()}")

# Vectorize text data with TF-IDF (adjusted parameters)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=2,  # Ignore terms that appear in less than 2 documents
    ngram_range=(1, 2),  # Use unigrams and bigrams
    max_features=5000  # Limit features to prevent overfitting
)
X_train_vectorized = vectorizer.fit_transform(X_train)

print(f"\nFeature matrix shape: {X_train_vectorized.shape}")

# Train the classifier with balanced class weights
clf = LinearSVC(
    class_weight='balanced',  # Automatically adjust weights
    C=1.0,  # Regularization parameter
    max_iter=2000,  # Increase max iterations
    random_state=42
)
clf.fit(X_train_vectorized, y_train)

# Evaluate on training set first (to check for overfitting)
train_preds = clf.predict(X_train_vectorized)
train_accuracy = clf.score(X_train_vectorized, y_train)
print(f"\nTraining accuracy: {train_accuracy:.4f}")
print(f"Training predictions distribution:\n{pd.Series(train_preds).value_counts()}")

# Evaluate on test set
X_test_vectorized = vectorizer.transform(X_test)
test_preds = clf.predict(X_test_vectorized)
test_accuracy = clf.score(X_test_vectorized, y_test)

print(f"\n{'='*50}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"\nTest predictions distribution:")
print(pd.Series(test_preds).value_counts())

print(f"\nActual test labels distribution:")
print(y_test.value_counts())

# Detailed classification report
print(f"\n{'='*50}")
print("Classification Report:")
print(classification_report(y_test, test_preds, target_names=['Real', 'Fake']))

# Confusion matrix
print(f"\n{'='*50}")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, test_preds)
print(f"              Predicted Real  Predicted Fake")
print(f"Actual Real        {cm[0][0]:5d}          {cm[0][1]:5d}")
print(f"Actual Fake        {cm[1][0]:5d}          {cm[1][1]:5d}")

# Save the trained model and vectorizer
joblib.dump(clf, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print(f"\n{'='*50}")
print("Model and vectorizer saved!")

# Function to test predictions on new text
def predict_news(text, web="", category=""):
    """Predict if news is fake or real"""
    input_text = f"{text} {web} {category}"
    vectorized = vectorizer.transform([input_text])
    prediction = clf.predict(vectorized)[0]
    
    # Get decision function scores for confidence
    decision = clf.decision_function(vectorized)[0]
    
    label = "FAKE" if prediction == 1 else "REAL"
    confidence = abs(decision)
    
    print(f"\nPrediction: {label}")
    print(f"Confidence score: {confidence:.4f}")
    return prediction

# Test with sample predictions
print(f"\n{'='*50}")
print("Sample Predictions:")
print("\nTest 1:")
predict_news("Breaking: Scientists discover cure for all diseases")
print("\nTest 2:")
predict_news("Local government announces new infrastructure project")