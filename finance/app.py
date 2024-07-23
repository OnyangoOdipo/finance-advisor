from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open(r'./finance/models/random_model.pkl', 'rb'))
scaler = pickle.load(open(r'./finance/models/scaler.pkl', 'rb'))
columns = pickle.load(open(r'./finance/models/columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('step1.html')

@app.route('/step2', methods=['POST'])
def step2():
    salary = request.form.get('salary')
    profession = request.form.get('profession')
    return render_template('step2.html', salary=salary, profession=profession)

@app.route('/step3', methods=['POST'])
def step3():
    salary = request.form.get('salary')
    profession = request.form.get('profession')
    housing = request.form.get('housing')
    food = request.form.get('food')
    transport = request.form.get('transport')
    return render_template('step3.html', salary=salary, profession=profession, housing=housing, food=food, transport=transport)

@app.route('/result', methods=['POST'])
def result():
    salary = float(request.form.get('salary'))
    profession = request.form.get('profession')
    housing = float(request.form.get('housing'))
    food = float(request.form.get('food'))
    transport = float(request.form.get('transport'))
    bills = float(request.form.get('bills'))
    clothing = float(request.form.get('clothing'))
    personal_needs = float(request.form.get('personal_needs'))
    debt_repayment = float(request.form.get('debt_repayment'))
    family_needs = float(request.form.get('family_needs'))
    health_insurance = float(request.form.get('health_insurance'))
    entertainment_leisure = float(request.form.get('entertainment_leisure'))
    expenditure = float(request.form.get('expenditure'))

    # Create a DataFrame for the input
    input_df = pd.DataFrame({
        'salary': [salary],
        'housing': [housing],
        'food': [food],
        'transport': [transport],
        'bills': [bills],
        'clothing': [clothing],
        'personal_needs': [personal_needs],
        'debt_repayment': [debt_repayment],
        'family_needs': [family_needs],
        'health_insurance': [health_insurance],
        'entertainment_leisure': [entertainment_leisure],
        'profession': [profession],
        'total_expenditure': [expenditure]
    })

    # Convert categorical features using get_dummies
    input_df = pd.get_dummies(input_df)
    
    # Align input data with the columns used during model training
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Generate financial advice
    advice = generate_financial_advice(expenditure, prediction)

    return render_template('result.html', prediction=prediction, advice=advice)

def generate_financial_advice(expenditure, prediction):
    advice = []
    if expenditure > prediction * 1.1:
        advice.append("Your expenditure is higher than expected. Consider reducing non-essential expenses.")
    elif expenditure < prediction * 0.9:
        advice.append("You are spending less than expected. You might have room for additional investments or savings.")
    else:
        advice.append("Your spending is within the expected range. Keep up the good work!")

    # Additional advice based on expenditure
    if expenditure > 0.3 * prediction:
        advice.append("Consider reviewing your housing and utility bills to find potential savings.")
    if expenditure > 0.2 * prediction:
        advice.append("Evaluate your food and transportation expenses for possible optimizations.")
    if expenditure > 0.1 * prediction:
        advice.append("Review your personal and family needs expenses to ensure they are within your budget.")

    return advice

if __name__ == '__main__':
    app.run(debug=True)
