from flask import Flask, render_template, request
import pickle
import pandas as pd
import re
import string
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Load the models
def load_models():
    with open('Logistic_regression.pkl', 'rb') as f:
        LR = pickle.load(f)
    with open('DecisionTreeClassifier.pkl', 'rb') as f:
        DT = pickle.load(f)
    with open('GradientBoostingClassifier.pkl', 'rb') as f:
        GBC = pickle.load(f)
    with open('RandomForestClassifier.pkl', 'rb') as f:
        RFC = pickle.load(f)
    return LR, DT, GBC, RFC

# Load the vectorizer
def load_vectorizer():
    with open('Vectorization.pkl', 'rb') as f:
        vectorization = pickle.load(f)
    return vectorization

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def get_gemini_analysis(news_text):
    prompt = f"""
    Analyze the following news article for authenticity. Consider:
    1. Credibility of sources
    2. Factual accuracy
    3. Tone and bias
    4. Presence of verifiable information
    
    News Article:
    {news_text}
    
    Provide a structured response with:
    1. Classification: (FAKE or REAL)
    2. Confidence Score: (0-100%)
    3. Key Indicators: (List main factors that led to this conclusion)
    4. Brief Explanation: (2-3 sentences)
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"

# Load models and vectorizer at startup
LR, DT, GBC, RFC = load_models()
vectorization = load_vectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        
        # Get Gemini analysis
        gemini_analysis = get_gemini_analysis(news)
        
        # Preprocess and predict with ML models
        testing_news = {"text":[news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt) 
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        
        # Get predictions from all models
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)

        predictions = {
            'Logistic Regression': output_label(pred_LR[0]),
            'Decision Tree': output_label(pred_DT[0]),
            'Gradient Boosting': output_label(pred_GBC[0]),
            'Random Forest': output_label(pred_RFC[0])
        }
        
        return render_template('result.html', 
                             predictions=predictions, 
                             news_text=news,
                             gemini_analysis=gemini_analysis)

if __name__ == '__main__':
    app.run(debug=True) 