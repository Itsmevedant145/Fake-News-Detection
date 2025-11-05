#!/usr/bin/env python3
"""
Quick Fix: Create Indian Context Metadata
Run this script to create the missing metadata file
"""

import joblib
import os

print("\n" + "="*70)
print(" CREATING INDIAN CONTEXT METADATA FOR IFND DATASET")
print("="*70 + "\n")

# Check if file already exists
if os.path.exists('indian_context_metadata.pkl'):
    response = input("⚠️  Metadata file already exists. Overwrite? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        exit()

# Trusted Indian News Sources (Expanded)
INDIAN_TRUSTED_SOURCES = [
    'times of india', 'toi', 'hindustan times', 'indian express', 
    'the hindu', 'ndtv', 'dnaindia', 'dna', 'zee news', 'aaj tak',
    'india today', 'firstpost', 'scroll', 'wire', 'quint',
    'news18', 'republic', 'hindustan', 'deccan herald', 'mint',
    'economic times', 'livemint', 'business standard', 'pti', 'ani',
    'thehindu', 'hindustantimes', 'indianexpress', 'indiatoday',
    'timesofindia', 'thewire', 'thequint', 'scroll.in',
    'moneycontrol', 'theprint', 'outlookindia', 'caravanmagazine',
    'bbc india', 'bbc hindi', 'reuters india', 'al jazeera india',
    'malayala manorama', 'mathrubhumi', 'dainik jagran', 'dainik bhaskar',
    'amar ujala', 'ananda bazar', 'eenadu', 'sakshi', 'dinamalar',
    'pib', 'mygov', 'who india', 'mohfw', 'dd news', 'doordarshan'
]

# Suspicious Source Indicators
SUSPICIOUS_SOURCES = [
    'unknown', 'whatsapp', 'facebook', 'twitter', 'social media',
    'viral', 'forward', 'broadcast', 'fake', 'exposed',
    'breaking', 'exclusive', 'truth', 'real', 'insider',
    'youtube', 'instagram', 'telegram', 'tiktok',
    'opindia', 'postcard', 'swarajya', 'fakingnews',
    'must watch', 'shocking', 'alert', 'warning', 'urgent'
]

# Sensational Keywords by Category
INDIAN_SENSATIONAL_KEYWORDS = {
    'political': [
        'modi', 'rahul', 'kejriwal', 'mamata', 'yogi', 'amit shah',
        'bjp', 'congress', 'aap', 'scam', 'corruption', 'expose',
        'anti-national', 'pakistan', 'china', 'surgical strike',
        'article 370', 'kashmir', 'caa', 'nrc', 'demonetization'
    ],
    'religious': [
        'hindu', 'muslim', 'christian', 'mandir', 'masjid', 'temple',
        'mosque', 'church', 'communal', 'riot', 'lynching',
        'love jihad', 'conversion', 'beef', 'jihad', 'hindutva'
    ],
    'covid': [
        'cure', 'vaccine', 'corona', 'covid', 'ayurveda', 'immunity',
        'lockdown', 'plasma', 'remdesivir', 'oxygen', 'aarogya setu',
        'covaxin', 'covishield', 'delta', 'omicron', 'black fungus'
    ],
    'sensational': [
        'shocking', 'unbelievable', 'breaking', 'exclusive', 'viral',
        'exposed', 'alert', 'warning', 'urgent', 'must watch',
        'truth', 'secret', 'leaked', 'caught', 'revelation'
    ],
    'implausible': [
        'alien', 'ufo', 'miracle', 'ghost', 'supernatural',
        'prophecy', 'immortal', 'teleport', 'time travel',
        'cure all', 'flat earth', '5g coronavirus', 'microchip'
    ]
}

# IFND Categories
IFND_CATEGORIES = [
    'COVID-19', 'ELECTION', 'POLITICS', 'TERROR', 'VIOLENCE',
    'SPORTS', 'ENTERTAINMENT', 'HEALTH', 'RELIGION'
]

# Create metadata dictionary
metadata = {
    'trusted_sources': INDIAN_TRUSTED_SOURCES,
    'suspicious_sources': SUSPICIOUS_SOURCES,
    'sensational_keywords': INDIAN_SENSATIONAL_KEYWORDS,
    'categories': IFND_CATEGORIES,
    'version': '1.0',
    'dataset': 'IFND'
}

# Save metadata
try:
    joblib.dump(metadata, 'indian_context_metadata.pkl')
    print("✅ SUCCESS! Metadata file created: indian_context_metadata.pkl\n")
    
    # Verify
    test_load = joblib.load('indian_context_metadata.pkl')
    print("✅ Verification: File loads correctly\n")
    
    # Show summary
    print("📊 Metadata Summary:")
    print(f"   • Trusted Sources: {len(INDIAN_TRUSTED_SOURCES)}")
    print(f"   • Suspicious Indicators: {len(SUSPICIOUS_SOURCES)}")
    print(f"   • Political Keywords: {len(INDIAN_SENSATIONAL_KEYWORDS['political'])}")
    print(f"   • Religious Keywords: {len(INDIAN_SENSATIONAL_KEYWORDS['religious'])}")
    print(f"   • COVID Keywords: {len(INDIAN_SENSATIONAL_KEYWORDS['covid'])}")
    print(f"   • Categories: {len(IFND_CATEGORIES)}")
    
    print("\n" + "="*70)
    print(" ✅ METADATA CREATED SUCCESSFULLY!")
    print(" You can now run your Flask API or training script")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}\n")
    print("Make sure you have joblib installed:")
    print("   pip install joblib")