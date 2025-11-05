import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier

import joblib
import re
import warnings
warnings.filterwarnings('ignore')

# Load dataset with better error handling
try:
    data = pd.read_csv('IFND.csv', encoding='iso-8859-1')
except FileNotFoundError:
    print("Error: IFND.csv not found!")
    exit()

print(f"Dataset loaded: {len(data)} rows")
print(f"Columns: {data.columns.tolist()}\n")

# Data cleaning and preprocessing
def clean_text(text):
    """Enhanced text cleaning"""
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Apply cleaning
print("Cleaning text data...")
data['Statement_clean'] = data['Statement'].apply(clean_text)
data['Web_clean'] = data['Web'].apply(clean_text)
data['Category_clean'] = data['Category'].apply(clean_text)

# Encode labels: 0 = Real, 1 = Fake
data['fake'] = data['Label'].str.upper().str.strip().map({'TRUE': 0, 'FAKE': 1})

# Remove rows with missing labels
data = data.dropna(subset=['fake'])
print(f"Data after removing invalid labels: {len(data)} rows")

# Check class distribution
print("\n" + "="*50)
print("Class Distribution:")
print(data['fake'].value_counts())
print(f"\nClass Balance:")
print(data['fake'].value_counts(normalize=True))
print(f"Imbalance ratio: {data['fake'].value_counts()[0] / data['fake'].value_counts()[1]:.2f}:1")

# Feature engineering: Create combined features
def create_features(row):
    """Create rich feature combinations"""
    features = []
    
    # Main statement (most important)
    features.append(row['Statement_clean'])
    
    # Add source credibility indicator
    features.append(f"source_{row['Web_clean']}")
    
    # Add category context
    features.append(f"category_{row['Category_clean']}")
    
    # Combine statement with category for context
    features.append(f"{row['Statement_clean']} in {row['Category_clean']}")
    
    return " ".join(features)

print("\nEngineering features...")
X = data.apply(create_features, axis=1)
y = data['fake']

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training class distribution:\n{y_train.value_counts()}")

# Advanced TF-IDF Vectorizer
print("\n" + "="*50)
print("Vectorizing text with advanced TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,          # Ignore terms in >70% of documents
    min_df=3,            # Ignore terms in <3 documents
    ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
    max_features=10000,  # Increased feature space
    sublinear_tf=True,   # Use log scaling for term frequency
    norm='l2'            # L2 normalization
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_vectorized.shape}")

# Handle class imbalance with SMOTE
print("\nApplying SMOTE for class balancing...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectorized, y_train)

print(f"Balanced training set size: {len(y_train_balanced)}")
print(f"Balanced class distribution:\n{pd.Series(y_train_balanced).value_counts()}")

# Train multiple models and create an ensemble
print("\n" + "="*50)
print("Training ensemble of classifiers...")

# Model 1: LinearSVC with tuned parameters
svc_model = LinearSVC(
    C=0.5,
    class_weight='balanced',
    max_iter=3000,
    random_state=42,
    dual=False  # More efficient for large datasets
)

# Model 2: Logistic Regression
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='saga'
)

# Model 3: Random Forest (works with sparse matrices)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Train individual models
print("Training LinearSVC...")
svc_model.fit(X_train_balanced, y_train_balanced)

print("Training Logistic Regression...")
lr_model.fit(X_train_balanced, y_train_balanced)

print("Training Random Forest...")
rf_model.fit(X_train_balanced, y_train_balanced)

# Evaluate individual models
print("\n" + "="*50)
print("Individual Model Performance on Test Set:\n")

models = {
    'LinearSVC': svc_model,
    'Logistic Regression': lr_model,
    'Random Forest': rf_model
}
# ===============================
# Train Gradient Boosting model
# ===============================
print("Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,       # number of trees
    learning_rate=0.1,      # step size for each tree
    max_depth=3,            # depth of each tree
    random_state=42
)
gb_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on test set
gb_acc = gb_model.score(X_test_vectorized, y_test)
gb_f1 = f1_score(y_test, gb_model.predict(X_test_vectorized))
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
print(f"Gradient Boosting F1-Score: {gb_f1:.4f}")

# Save the model
joblib.dump(gb_model, 'fake_news_gb.pkl')
print("✅ Saved Gradient Boosting model as fake_news_gb.pkl")


test_predictions = {}
for name, model in models.items():
    preds = model.predict(X_test_vectorized)
    acc = model.score(X_test_vectorized, y_test)
    f1 = f1_score(y_test, preds)
    test_predictions[name] = preds
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print()

# Ensemble voting (majority vote)
print("="*50)
print("Ensemble Prediction (Majority Voting):\n")

ensemble_preds = np.round(
    (test_predictions['LinearSVC'] + 
     test_predictions['Logistic Regression'] + 
     test_predictions['Random Forest']) / 3
).astype(int)

ensemble_accuracy = (ensemble_preds == y_test).mean()
ensemble_f1 = f1_score(y_test, ensemble_preds)

print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble F1-Score: {ensemble_f1:.4f}")

# Detailed classification report for best model
print("\n" + "="*50)
print("Detailed Classification Report (Ensemble):")
print(classification_report(y_test, ensemble_preds, target_names=['Real', 'Fake']))

# Confusion matrix
print("="*50)
print("Confusion Matrix (Ensemble):")
cm = confusion_matrix(y_test, ensemble_preds)
print(f"\n              Predicted Real  Predicted Fake")
print(f"Actual Real        {cm[0][0]:5d}          {cm[0][1]:5d}")
print(f"Actual Fake        {cm[1][0]:5d}          {cm[1][1]:5d}")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nAdditional Metrics:")
print(f"True Positives (Fake detected as Fake): {tp}")
print(f"True Negatives (Real detected as Real): {tn}")
print(f"False Positives (Real detected as Fake): {fp}")
print(f"False Negatives (Fake detected as Real): {fn}")
print(f"Precision (Fake): {tp/(tp+fp):.4f}")
print(f"Recall (Fake): {tp/(tp+fn):.4f}")

# Save all models and vectorizer
print("\n" + "="*50)
print("Saving models...")

joblib.dump(svc_model, 'fake_news_svc.pkl')
joblib.dump(lr_model, 'fake_news_lr.pkl')
joblib.dump(rf_model, 'fake_news_rf.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_v2.pkl')

print("All models and vectorizer saved!")

# Enhanced prediction function
def predict_news(statement, web="", category="", show_confidence=True):
    """
    Predict if news is fake or real using ensemble
    
    Args:
        statement: News statement text
        web: Source website
        category: News category
        show_confidence: Whether to show confidence scores
    """
    # Clean inputs
    statement_clean = clean_text(statement)
    web_clean = clean_text(web)
    category_clean = clean_text(category)
    
    # Create features
    input_text = (f"{statement_clean} source_{web_clean} "
                  f"category_{category_clean} {statement_clean} in {category_clean}")
    
    # Vectorize
    vectorized = vectorizer.transform([input_text])
    
    # Get predictions from all models
    svc_pred = svc_model.predict(vectorized)[0]
    lr_pred = lr_model.predict(vectorized)[0]
    rf_pred = rf_model.predict(vectorized)[0]
    
    # Ensemble prediction (majority vote)
    predictions = [svc_pred, lr_pred, rf_pred]
    final_pred = int(np.round(np.mean(predictions)))
    
    # Calculate confidence (agreement level)
    agreement = sum(predictions) / len(predictions)
    confidence = max(agreement, 1 - agreement)
    
    label = "⚠️  FAKE NEWS" if final_pred == 1 else "✓ REAL NEWS"
    
    print(f"\nStatement: {statement[:100]}...")
    print(f"Source: {web}")
    print(f"Category: {category}")
    print(f"\n{'-'*50}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.1f}%")
    
    if show_confidence:
        print(f"\nModel Votes:")
        print(f"  LinearSVC: {'FAKE' if svc_pred == 1 else 'REAL'}")
        print(f"  Logistic Regression: {'FAKE' if lr_pred == 1 else 'REAL'}")
        print(f"  Random Forest: {'FAKE' if rf_pred == 1 else 'REAL'}")
    
    return final_pred, confidence

# Test with sample predictions
print("\n" + "="*50)
print("Sample Predictions:\n")

test_cases = [
    ("Breaking: Scientists discover cure for all diseases overnight!", "Unknown", "COVID-19"),
    ("WHO praises India's Aarogya Setu app, says it helped in identifying COVID-19 clusters", "DNAINDIA", "COVID-19"),
    ("Aliens land in New York City, government confirms", "FakeNewsDaily", "POLITICS"),
    ("Local government announces new infrastructure project", "TimesOfIndia", "POLITICS"),
    ("Drinking bleach cures coronavirus, says expert", "Unknown", "COVID-19")
]

for i, (statement, web, category) in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    predict_news(statement, web, category)
    print("="*50)

print("\n✅ Model training and evaluation complete!")