import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Step 1: Load data
DF_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_dataset.csv')
df = pd.read_csv(DF_PATH)
print("Data loaded:", df.shape)

# Step 2: Encode text columns to numbers
le_industry = LabelEncoder()
le_ethnicity = LabelEncoder()
le_citizen = LabelEncoder()

df['Industry'] = le_industry.fit_transform(df['Industry'])
df['Ethnicity'] = le_ethnicity.fit_transform(df['Ethnicity'])
df['Citizen'] = le_citizen.fit_transform(df['Citizen'])

# Step 3: Separate features and target
X = df.drop('Approved', axis=1)
y = df['Approved']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale numbers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Test model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Save everything
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
pickle.dump(model, open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb'))
pickle.dump(scaler, open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb'))
pickle.dump(le_industry, open(os.path.join(MODEL_DIR, 'le_industry.pkl'), 'wb'))
pickle.dump(le_ethnicity, open(os.path.join(MODEL_DIR, 'le_ethnicity.pkl'), 'wb'))
pickle.dump(le_citizen, open(os.path.join(MODEL_DIR, 'le_citizen.pkl'), 'wb'))

print("Model saved successfully!")
