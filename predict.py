import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("=== 2D Prediction System ===\n")

# Load model
print("Loading model...")
with open('model_range.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data with flexible date parsing
print("Loading data...")
df = pd.read_csv('combined_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
df = df.dropna(subset=['Date', '2D'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Total records loaded: {len(df)}")

# Get recent results
recent = df.tail(10)['2D'].values
print(f"\nRecent 10 results: {recent}")

# Prepare features
recent_5 = recent[-5:]
features = {
    'year': 2025,
    'month': 10,
    'day': 22,
    'day_of_week': 2,
    'prev_1': float(recent_5[-1]),
    'prev_2': float(recent_5[-2]),
    'prev_3': float(recent_5[-3]),
    'prev_4': float(recent_5[-4]),
    'prev_5': float(recent_5[-5]),
    'rolling_mean_5': float(np.mean(recent_5)),
    'rolling_std_5': float(np.std(recent_5))
}

X = pd.DataFrame([features])

# Predict
pred = model.predict(X)[0]
proba = model.predict_proba(X)[0]

range_names = ['0-24', '25-49', '50-74', '75-99']

print(f"\n{'='*50}")
print(f"TODAY'S PREDICTION - {datetime.now().strftime('%Y-%m-%d')}")
print(f"{'='*50}")

print(f"\n🎯 Predicted Range: {range_names[pred]}")
print(f"\n📊 Range Probabilities:")
for i, p in enumerate(proba):
    bar = '█' * int(p * 40)
    print(f"  {range_names[i]:8s}: {p:5.1%} {bar}")

# Top 10 in predicted range
ranges = [(0,25), (25,50), (50,75), (75,100)]
start, end = ranges[pred]
nums = list(range(start, end))

print(f"\n✅ Top 10 recommendations: {nums[:10]}")
print(f"\n{'='*50}")

