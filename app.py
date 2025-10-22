from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and data
try:
    model = joblib.load('model_range.pkl')
    df = pd.read_csv('combined_data.csv')
    print("✅ Model and data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/data: {e}")

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Get 2D predictions for selected date"""
    try:
        # Get date from request
        data = request.get_json()
        selected_date = data.get('date')
        
        if not selected_date:
            return jsonify({
                'success': False,
                'error': 'Date is required'
            }), 400
        
        # Parse date
        date_obj = pd.to_datetime(selected_date)
        
        # Get recent 2D values from dataset
        recent_values = df['2D'].tail(10).values
        
        # Create 11 features for prediction (model expects 11 features)
        feature_array = [
            date_obj.year,              # 1. year
            date_obj.month,             # 2. month
            date_obj.day,               # 3. day
            date_obj.dayofweek,         # 4. day_of_week
            float(recent_values[-1]),   # 5. prev_1
            float(recent_values[-2]),   # 6. prev_2
            float(recent_values[-3]),   # 7. prev_3
            float(recent_values[-4]),   # 8. prev_4
            float(recent_values[-1]),   # 9. prev_1_float (duplicate for compatibility)
            float(recent_values[-2]),   # 10. prev_2_float
            float(recent_values[-3])    # 11. prev_3_float
        ]
        
        print(f"📊 Features: {len(feature_array)} features")
        print(f"📊 Feature values: {feature_array}")
        
        # Get predictions from model
        probabilities = model.predict_proba([feature_array])[0]
        
        # Create prediction results
        predictions = []
        for number in range(100):  # 00-99
            predictions.append({
                'number': f"{number:02d}",
                'probability': float(probabilities[number])
            })
        
        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Return top 10 predictions
        return jsonify({
            'success': True,
            'predictions': predictions[:10],
            'date': selected_date
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': 'model' in globals(),
        'data_loaded': 'df' in globals()
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)




