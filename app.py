from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model and tools
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
le_industry = pickle.load(open('models/le_industry.pkl', 'rb'))
le_ethnicity = pickle.load(open('models/le_ethnicity.pkl', 'rb'))
le_citizen = pickle.load(open('models/le_citizen.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    debt = float(request.form['debt'])
    married = int(request.form['married'])
    bank_customer = int(request.form['bank_customer'])
    industry = request.form['industry']
    ethnicity = request.form['ethnicity']
    years_employed = float(request.form['years_employed'])
    prior_default = int(request.form['prior_default'])
    employed = int(request.form['employed'])
    credit_score = int(request.form['credit_score'])
    drivers_license = int(request.form['drivers_license'])
    citizen = request.form['citizen']
    zipcode = int(request.form['zipcode'])
    income = int(request.form['income'])
    
    # Encode text values
    industry_encoded = le_industry.transform([industry])[0]
    ethnicity_encoded = le_ethnicity.transform([ethnicity])[0]
    citizen_encoded = le_citizen.transform([citizen])[0]
    
    # Create input array
    input_data = np.array([[gender, age, debt, married, bank_customer, 
                           industry_encoded, ethnicity_encoded, years_employed,
                           prior_default, employed, credit_score, drivers_license,
                           citizen_encoded, zipcode, income]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    result = "APPROVED" if prediction == 1 else "REJECTED"
    confidence = probability[prediction] * 100
    print("Prediction:", prediction)
    print("Probability:", probability)
    print("Input:", input_data)
    
    return render_template('result.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
