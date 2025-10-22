from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('model_range.pkl')
    df = pd.read_csv('combined_data.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date', '2D']).sort_values('Date').reset_index(drop=True)
    print(f"✅ Loaded: {len(df)} rows, {model.n_classes_} classes")
except Exception as e:
    print(f"❌ Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        date = data.get('date')
        if not date: return jsonify({'success': False, 'error': 'Date required'}), 400
        
        date_obj = pd.to_datetime(date)
        if len(df) < 10: return jsonify({'success': False, 'error': f'Need 10+ rows'}), 400
        
        recent = df['2D'].tail(10).values
        features = [date_obj.year, date_obj.month, date_obj.day, date_obj.dayofweek,
                   float(recent[-1]), float(recent[-2]), float(recent[-3]), float(recent[-4]),
                   float(recent[-1]), float(recent[-2]), float(recent[-3])]
        
        probas = model.predict_proba([features])[0]
        
        if len(probas) == 4:
            ranges = ['00-24', '25-49', '50-74', '75-99']
            preds = [{'range': ranges[i], 'probability': float(probas[i]), 
                     'percentage': f"{probas[i]:.1%}"} for i in range(4)]
            preds.sort(key=lambda x: x['probability'], reverse=True)
            return jsonify({'success': True, 'predictions': preds, 'date': date, 
                          'recent': [int(x) for x in recent[-5:]]})
        else:
            preds = [{'number': f"{i:02d}", 'probability': float(probas[i]), 
                     'percentage': f"{probas[i]:.1%}"} for i in range(100)]
            preds.sort(key=lambda x: x['probability'], reverse=True)
            return jsonify({'success': True, 'predictions': preds[:10], 'date': date,
                          'recent': [int(x) for x in recent[-5:]]})
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'rows': len(df), 'classes': model.n_classes_})

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))









