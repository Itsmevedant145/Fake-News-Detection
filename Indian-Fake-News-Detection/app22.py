from flask import Flask, request, jsonify
from flask_cors import CORS ,cross_origin
import joblib
import numpy as np
import re
import traceback



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
print("Loading models for Indian Fake News Detection...")
try:
    svc_model = joblib.load('fake_news_svc.pkl')
    lr_model = joblib.load('fake_news_lr.pkl')
    gb_model = joblib.load('fake_news_gb.pkl')
    vectorizer = joblib.load('tfidf_vectorizer_v2.pkl')
    metadata = joblib.load('indian_context_metadata.pkl')
    
    INDIAN_TRUSTED_SOURCES = metadata['trusted_sources']
    SUSPICIOUS_SOURCES = metadata['suspicious_sources']
    INDIAN_SENSATIONAL_KEYWORDS = metadata['sensational_keywords']
    
    print("✅ All models and metadata loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# ============================================================================
# HELPER FUNCTIONS (Same as training script)
# ============================================================================
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

def clean_text(text):
    """Enhanced text cleaning for Indian context"""
    if not text or text.strip() == "":
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    text = ' '.join(text.split())
    return text

def get_source_credibility(source):
    """Calculate source credibility score for Indian sources"""
    if not source or source.strip() == "":
        return 0.0
    
    source_lower = str(source).lower()
    
    for trusted in INDIAN_TRUSTED_SOURCES:
        if trusted in source_lower:
            return 1.0
    
    for suspicious in SUSPICIOUS_SOURCES:
        if suspicious in source_lower:
            return -1.0
    
    return 0.0

def extract_sensationalism_score(text):
    """Calculate sensationalism score based on Indian context"""
    if not text or text.strip() == "":
        return 0
    
    text_lower = str(text).lower()
    score = 0
    
    for category, keywords in INDIAN_SENSATIONAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
    
    score += text.count('!') * 0.5
    score += text.count('?') * 0.3
    
    words = text.split()
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    score += caps_words * 0.5
    
    return min(score, 10)

def check_implausible_claims(text):
    """Detect obviously implausible claims"""
    if not text or text.strip() == "":
        return False
    
    text_lower = str(text).lower()
    
    implausible_patterns = [
        r'alien.*land', r'ufo.*sight', r'cure.*all.*disease',
        r'immortal', r'teleport', r'time.*travel',
        r'celebrity.*died.*fake', r'government.*confirm.*alien',
        r'scientist.*prove.*(god|supernatural)', r'miracle.*cure',
        r'drink.*bleach', r'5g.*coronavirus', r'bill.*gates.*microchip',
        r'vaccine.*kill', r'earth.*flat', r'reptilian'
    ]
    
    for pattern in implausible_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def create_enhanced_features(statement, web, category):
    """Create rich features with Indian context"""
    statement_clean = clean_text(statement)
    web_clean = clean_text(web)
    category_clean = clean_text(category)
    
    credibility = get_source_credibility(web)
    credibility_tag = "trusted" if credibility > 0 else ("suspicious" if credibility < 0 else "unknown")
    
    sensational_score = extract_sensationalism_score(statement)
    sensational_tag = f"sensational_{min(int(sensational_score), 5)}"
    
    implausible = check_implausible_claims(statement)
    implausible_tag = "implausible" if implausible else "plausible"
    
    features = [
        statement_clean,
        f"source_{web_clean}",
        f"category_{category_clean}",
        f"credibility_{credibility_tag}",
        f"sensationalism_{sensational_tag}",
        f"claim_{implausible_tag}",
        f"{statement_clean} in {category_clean}",
        f"{category_clean} from {web_clean}"
    ]
    
    return " ".join(features)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.json
        
        # Get inputs
        text = data.get('text', '').strip()
        web = data.get('web', '').strip()
        category = data.get('category', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Calculate features
        credibility = get_source_credibility(web)
        sensationalism = extract_sensationalism_score(text)
        implausible = check_implausible_claims(text)
        
        # Create feature string
        input_features = create_enhanced_features(text, web, category)
        
        # Vectorize
        vectorized_text = vectorizer.transform([input_features])
        
        # Get predictions from all models
        svc_pred = svc_model.predict(vectorized_text)[0]
        lr_pred = lr_model.predict(vectorized_text)[0]
        gb_pred = gb_model.predict(vectorized_text)[0]
        
        # Get decision scores
        try:
            svc_score = abs(svc_model.decision_function(vectorized_text)[0])
        except:
            svc_score = 1.0
            
        try:
            lr_score = abs(lr_model.decision_function(vectorized_text)[0])
        except:
            lr_score = 1.0
        
        try:
            gb_proba = gb_model.predict_proba(vectorized_text)[0]
            gb_score = max(gb_proba)
        except:
            gb_score = 1.0
        
        # Ensemble prediction
        predictions = [svc_pred, lr_pred, gb_pred]
        final_prediction = int(np.round(np.mean(predictions)))
        
        # Calculate confidence
        agreement = sum(predictions) / len(predictions)
        confidence_ratio = max(agreement, 1 - agreement)
        
        avg_decision_score = (svc_score + lr_score + gb_score) / 3
        combined_confidence = (confidence_ratio * 0.6) + (min(avg_decision_score / 2, 1.0) * 0.4)
        confidence_percentage = int(combined_confidence * 100)
        
        # Override for implausible claims
        override_applied = False
        override_reason = None
        
        if implausible and final_prediction == 0 and confidence_percentage < 95:
            final_prediction = 1
            confidence_percentage = 75
            override_applied = True
            override_reason = "Implausible claim detected"
        
        # AGGRESSIVE: Override for highly suspicious sources (WhatsApp, Facebook)
        highly_suspicious = ['whatsapp', 'facebook', 'twitter', 'viral', 'forward']
        if any(sus in web.lower() for sus in highly_suspicious) and category.upper() in ['COVID-19', 'HEALTH', 'POLITICS']:
            # If suspicious source + sensitive category, be more aggressive
            if final_prediction == 0 and confidence_percentage < 70:
                final_prediction = 1
                confidence_percentage = 65
                override_applied = True
                override_reason = f"Highly suspicious source ({web}) for sensitive topic"
        
        # Result label
        result = 'Fake' if final_prediction == 1 else 'Real'
        
        # Model votes
        model_votes = {
            'LinearSVC': 'Fake' if svc_pred == 1 else 'Real',
            'Logistic Regression': 'Fake' if lr_pred == 1 else 'Real',
            'Gradient Boosting': 'Fake' if gb_pred == 1 else 'Real'
        }
        
        # Generate detailed reason with Indian context
        credibility_text = "TRUSTED" if credibility > 0 else ("SUSPICIOUS" if credibility < 0 else "UNKNOWN")
        
        if final_prediction == 1:
            if confidence_percentage >= 85:
                reason = f"🚨 HIGH CONFIDENCE FAKE NEWS. All or most models agree this is misinformation. "
            elif confidence_percentage >= 70:
                reason = f"⚠️ LIKELY FAKE NEWS. Multiple models indicate this may be false. "
            else:
                reason = f"⚡ POSSIBLY FAKE NEWS. Models show mixed signals. "
            
            if credibility < 0:
                reason += f"The source is flagged as SUSPICIOUS. "
            elif credibility == 0:
                reason += f"The source is UNKNOWN or UNVERIFIED. "
            
            if sensationalism > 5:
                reason += f"High sensationalism detected (score: {sensationalism:.1f}/10). "
            
            if implausible:
                reason += f"Contains IMPLAUSIBLE CLAIMS that are commonly found in fake news. "
            
            reason += "Please verify with trusted Indian news sources before sharing."
            
        else:
            if confidence_percentage >= 85:
                reason = f"✅ HIGH CONFIDENCE REAL NEWS. All or most models agree this appears credible. "
            elif confidence_percentage >= 70:
                reason = f"✓ LIKELY REAL NEWS. Multiple indicators suggest this is legitimate. "
            else:
                reason = f"🤔 POSSIBLY REAL NEWS. Models show mixed signals. "
            
            if credibility > 0:
                reason += f"Source is a TRUSTED Indian news outlet. "
            
            if sensationalism < 3:
                reason += f"Low sensationalism (score: {sensationalism:.1f}/10). "
            
            reason += "However, always cross-verify important news with multiple sources."
        
        # Analysis notes
        analysis_notes = []
        
        if web:
            analysis_notes.append(f"Source: {web} (Credibility: {credibility_text})")
        
        if category:
            analysis_notes.append(f"Category: {category}")
        
        analysis_notes.append(f"Sensationalism Score: {sensationalism:.1f}/10")
        
        if implausible:
            analysis_notes.append("⚠️ Contains implausible claims")
        
        vote_count = sum(predictions)
        if vote_count == 3:
            analysis_notes.append("Unanimous: All 3 models say FAKE")
        elif vote_count == 0:
            analysis_notes.append("Unanimous: All 3 models say REAL")
        else:
            analysis_notes.append(f"Split: {vote_count} FAKE, {3-vote_count} REAL")
        
        if override_applied:
            analysis_notes.append(f"🛡️ Override applied: {override_reason}")
        
        # Response
        response = {
            'prediction': result,
            'confidence': f"{confidence_percentage}/100",
            'confidence_percentage': confidence_percentage,
            'reason': reason,
            'model_votes': model_votes,
            'model_agreement': f"{int(confidence_ratio * 100)}%",
            'analysis_notes': analysis_notes,
            'indian_context': {
                'source_credibility': credibility_text,
                'sensationalism_score': round(sensationalism, 1),
                'has_implausible_claims': implausible,
                'override_applied': override_applied
            },
            'decision_score': float(avg_decision_score),
            'metadata': {
                'ensemble_used': True,
                'models_count': 3,
                'input_length': len(text),
                'dataset': 'IFND (Indian Fake News Dataset)'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'version': '3.0-indian-context',
        'dataset': 'IFND'
    })

@app.route('/sources', methods=['GET'])
@cross_origin()
def get_sources():
    """Get list of trusted and suspicious sources"""
    return jsonify({
        'trusted_sources': INDIAN_TRUSTED_SOURCES[:20],  # First 20
        'suspicious_indicators': SUSPICIOUS_SOURCES,
        'total_trusted': len(INDIAN_TRUSTED_SOURCES),
        'note': 'Based on IFND dataset and Indian media landscape'
    })

@app.route('/models-info', methods=['GET'])
@cross_origin()
def models_info():
    """Get information about loaded models"""
    return jsonify({
        'models': [
            'LinearSVC (C=0.5, balanced)',
            'Logistic Regression (saga solver)',
            'Gradient Boosting (100 estimators)'
        ],
        'ensemble_method': 'Majority Voting with Override',
        'vectorizer': 'TF-IDF (ngrams 1-3, 12k features)',
        'features': [
            'Statement text',
            'Source website with credibility scoring',
            'News category',
            'Sensationalism detection',
            'Implausible claim detection',
            'Contextual combinations'
        ],
        'dataset': 'IFND (Indian Fake News Dataset)',
        'dataset_size': '5,182 fact-checked articles',
        'date_range': '2013-2021',
        'special_features': [
            'Indian source credibility scoring',
            'Indian political/religious context awareness',
            'WhatsApp/social media forward detection',
            'COVID-19 misinformation patterns',
            'Implausible claim override'
        ]
    })

@app.route('/test-examples', methods=['GET'])
@cross_origin()
def test_examples():
    """Get example news for testing"""
    return jsonify({
        'examples': [
            {
                'text': 'WHO praises India\'s Aarogya Setu app, says it helped in identifying COVID-19 clusters',
                'web': 'DNAINDIA',
                'category': 'COVID-19',
                'expected': 'Real'
            },
            {
                'text': 'Scientists confirm aliens landed in New York City',
                'web': 'Unknown',
                'category': 'SCIENCE',
                'expected': 'Fake'
            },
            {
                'text': 'WhatsApp forward claims drinking cow urine cures coronavirus',
                'web': 'WhatsApp',
                'category': 'COVID-19',
                'expected': 'Fake'
            },
            {
                'text': 'Prime Minister announces new economic reforms',
                'web': 'Times of India',
                'category': 'POLITICS',
                'expected': 'Real'
            },
            {
                'text': 'Breaking: 5G towers spreading COVID-19, government confirms',
                'web': 'ViralNews',
                'category': 'COVID-19',
                'expected': 'Fake'
            }
        ]
    })

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Indian Fake News Detection API - IFND Enhanced',
        'version': '3.0',
        'dataset': 'IFND (Indian Fake News Dataset)',
        'endpoints': {
            'POST /predict': {
                'description': 'Predict if Indian news is fake or real',
                'parameters': {
                    'text': 'News statement (required)',
                    'web': 'Source website (optional but recommended)',
                    'category': 'News category (optional)'
                },
                'example': {
                    'text': 'WHO praises India Aarogya Setu app',
                    'web': 'DNAINDIA',
                    'category': 'COVID-19'
                }
            },
            'GET /health': 'Check API health status',
            'GET /models-info': 'Get detailed model information',
            'GET /sources': 'Get trusted and suspicious sources',
            'GET /test-examples': 'Get example news for testing'
        },
        'features': [
            'Indian context awareness',
            'Source credibility scoring for Indian media',
            'Sensationalism detection',
            'Implausible claim detection',
            'Ensemble of 3 models',
            'SMOTE-balanced training'
        ]
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" INDIAN FAKE NEWS DETECTION API")
    print(" Dataset: IFND (2013-2021)")
    print(" Ready to serve predictions!")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)