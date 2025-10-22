import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

print("=== STEP 2: Training 2D Prediction Model ===\n")

# 1. Load data
print("1. Loading data...")
df = pd.read_csv('combined_data.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', '2D'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"   Total records: {len(df)}")

# 2. Create range labels
print("\n2. Creating range labels...")
df['range'] = pd.cut(df['2D'], bins=[0, 25, 50, 75, 100], 
                      labels=[0, 1, 2, 3], right=False)

# 3. Create features
print("3. Creating features...")
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek

# Previous results
for i in range(1, 6):
    df[f'prev_{i}'] = df['2D'].shift(i)

# Rolling statistics
df['rolling_mean_5'] = df['2D'].rolling(5).mean()
df['rolling_std_5'] = df['2D'].rolling(5).std()

# Remove NaN
df = df.dropna()

# 4. Prepare training data
print("4. Preparing training data...")
feature_cols = ['year', 'month', 'day', 'day_of_week', 
                'prev_1', 'prev_2', 'prev_3', 'prev_4', 'prev_5',
                'rolling_mean_5', 'rolling_std_5']

X = df[feature_cols]
y = df['range']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# 5. Train model
print("\n5. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
print("\n6. Evaluating...")
accuracy = model.score(X_test, y_test)
print(f"   Accuracy: {accuracy:.2%}")

# 7. Save model
print("\n7. Saving model...")
with open('model_range.pkl', 'wb') as f:
    pickle.dump(model, f)

print("   ✅ Model saved: model_range.pkl")

# 8. Test prediction
print("\n8. Testing prediction...")
test_sample = X_test.iloc[0:1]
pred = model.predict(test_sample)[0]
pred_proba = model.predict_proba(test_sample)[0]

range_names = ['0-24', '25-49', '50-74', '75-99']
print(f"   Predicted range: {range_names[pred]}")
print(f"   Probabilities: {pred_proba}")

print("\n✅ STEP 2 COMPLETE! Model trained successfully!")
