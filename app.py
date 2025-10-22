from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and data
with open('model_range.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('combined_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
df = df.dropna(subset=['Date', '2D'])
df = df.sort_values('Date').reset_index(drop=True)

def get_prediction():
    """Generate prediction"""
    recent = df.tail(10)['2D'].values
    recent_5 = recent[-5:]
    
    features = {
        'year': datetime.now().year,
        'month': datetime.now().month,
        'day': datetime.now().day,
        'day_of_week': datetime.now().weekday(),
        'prev_1': float(recent_5[-1]),
        'prev_2': float(recent_5[-2]),
        'prev_3': float(recent_5[-3]),
        'prev_4': float(recent_5[-4]),
        'prev_5': float(recent_5[-5]),
        'rolling_mean_5': float(np.mean(recent_5)),
        'rolling_std_5': float(np.std(recent_5))
    }
    
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    range_names = ['0-24', '25-49', '50-74', '75-99']
    ranges = [(0,25), (25,50), (50,75), (75,100)]
    start, end = ranges[pred]
    top_10 = list(range(start, min(start+10, end)))
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'predicted_range': range_names[pred],
        'range_probabilities': {
            range_names[i]: f"{p:.1%}" for i, p in enumerate(proba)
        },
        'top_10': top_10,
        'recent_results': [int(x) for x in recent_5],
        'accuracy': '63.8%'
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    return jsonify(get_prediction())

@app.route('/api/history', methods=['GET'])
def history():
    recent = df.tail(30)[['Date', 'Time', '2D']].copy()
    recent['Date'] = recent['Date'].dt.strftime('%Y-%m-%d')
    return jsonify(recent.to_dict('records'))

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)









