import pickle
import re
import string
from flask import Flask, request, jsonify
from scipy.sparse import hstack
from textstat import flesch_reading_ease
from nltk.sentiment import SentimentIntensityAnalyzer
from flask_cors import CORS  # Import CORS
# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Load the trained model, vectorizer, and sentiment analyzer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('sentiment_analyzer.pkl', 'rb') as sid_file:
    sid = pickle.load(sid_file)

# Clean and preprocess the text
def clean_text(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", '', text.lower())

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    # Clean the input text
    text_cleaned = clean_text(text)
    
    # Vectorize the input text
    tfidf_text = vectorizer.transform([text_cleaned])

    # Calculate sentiment and readability scores
    sentiment_score = sid.polarity_scores(text_cleaned)['compound']
    readability_score = flesch_reading_ease(text_cleaned)

    # Combine all features
    features = hstack((tfidf_text, [[sentiment_score, readability_score]]))

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction result
    result = "real" if prediction[0] == 1 else "fake"
    return jsonify({'prediction': result})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # You can specify the port as needed

