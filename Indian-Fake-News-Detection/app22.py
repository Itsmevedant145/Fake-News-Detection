from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')  # expecting "text" key

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    
    # Get decision function scores for confidence
    decision_score = model.decision_function(vectorized_text)[0]
    
    # Calculate confidence as a percentage (normalize the decision score)
    confidence_raw = abs(decision_score)
    # Scale confidence to 0-100 range (you can adjust the scaling factor)
    confidence_percentage = min(100, int(confidence_raw * 10))
    
    result = 'Fake' if prediction == 1 else 'Real'
    
    # Generate reason based on prediction
    if prediction == 1:
        reason = "The search results and analysis indicate that this content may contain misleading or unverified information. Our AI model has detected patterns commonly associated with fake news."
    else:
        reason = "The content appears to be credible based on our analysis. The information aligns with verified sources and shows patterns consistent with legitimate news."
    
    return jsonify({
        'prediction': result,
        'confidence': f"{confidence_percentage}/100",
        'reason': reason,
        'confidence_score': float(confidence_raw)
    })

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch.nn.functional as F

# app = Flask(__name__)
# CORS(app)

# # Load the fine-tuned BERT model and tokenizer
# model_name_or_path = "bert-fake-news"
# tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
# model = BertForSequenceClassification.from_pretrained(model_name_or_path)
# model.eval()  # Set model to evaluation mode

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     text = data.get('text', '')

#     if not text:
#         return jsonify({'error': 'No text provided'}), 400

#     # Tokenize input text
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

#     # Get model outputs (logits)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     # Convert logits to probabilities (optional)
#     probs = F.softmax(logits, dim=1)
#     confidence, predicted_class = torch.max(probs, dim=1)

#     # Map predicted class to label
#     label_map = {0: "Real", 1: "Fake"}
#     prediction = label_map[predicted_class.item()]

#     return jsonify({
#         'prediction': prediction,
#         'confidence': confidence.item()
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
