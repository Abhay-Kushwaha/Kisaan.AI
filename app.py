from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import pandas as pd
from disease import predict_disease
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crop_price', methods=['GET', 'POST'])
def crop_price():
    df = pd.read_csv("dataset/Crop_Price.csv")
    state_encoder = pickle.load(open("model/state_encoder.pkl", "rb"))
    crop_encoder = pickle.load(open("model/crop_encoder.pkl", "rb"))
    crop_price_model = pickle.load(open("model/crop_price_model.pkl", "rb"))
    unique_states = sorted(df['State'].unique())
    unique_crops = sorted(df['Crop'].unique())
    if request.method == 'POST':
        try:
            state = request.form['State']
            crop = request.form['Crop']
            cost1 = float(request.form['CostCultivation']) #Crop amount in per hectare
            cost2 = float(request.form['CostCultivation2']) # Labour cost
            production = float(request.form['Production'])
            yield_amt = float(request.form['Yield'])
            temperature = float(request.form['Temperature'])
            rainfall = float(request.form['RainFall_Annual'])
            encoded_state = state_encoder.transform([state])[0]
            encoded_crop = crop_encoder.transform([crop])[0]
            features = np.array([[encoded_state, encoded_crop, cost1, cost2, production, yield_amt, temperature, rainfall]])
            prediction = crop_price_model.predict(features)[0]
            return render_template('crop_price.html', prediction=round(prediction, 2), states=unique_states, crops=unique_crops)
        except Exception as e:
            return render_template('crop_price.html', prediction=f"Error: {str(e)}", states=unique_states, crops=unique_crops)
    return render_template('crop_price.html',
                           states=unique_states,
                           crops=unique_crops)

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    crop_model = pickle.load(open("model/crop_predict_model.pkl", "rb"))
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = crop_model.predict([np.array(features)])
        return render_template('crop.html', prediction=prediction[0])
    return render_template('crop.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    with open("dataset/symptoms.json", "r") as f:
        symptoms_list = json.load(f)
    if request.method == 'POST':
        user_symptoms = request.form.getlist("symptoms[]")  # Get symptoms as a list
        days = int(request.form.get("days", 5))
        if not user_symptoms:
            return jsonify({"error": "No symptoms entered"}), 400
        advice, predictions = predict_disease(user_symptoms, days)
        response = {
            "advice": advice,
            "predictions": []
        }
        for disease, details in predictions.items():
            response["predictions"].append({
                "disease": disease,
                "description": details["desc"],
                "precautions": details["prec"],
                "medications": details["drugs"]["Medications"],
                "diet": details["drugs"]["Diet"]
            })
        return jsonify(response)
    return render_template('disease.html', symptoms=symptoms_list)

@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    breast_cancer_model = pickle.load(open("model/breast_cancer_model.pkl", "rb"))
    features = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
    'mean_smoothness', 'mean_compactness', 'mean_concavity',
    'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension'
]
    values = []
    result = None
    if request.method == 'POST':
        values = [float(request.form.get(f)) for f in features]
        prediction = breast_cancer_model.predict([values])[0]
        result = "Malignant (Cancer)" if prediction == 0 else "Benign (Non-Cancerous)"
    return render_template('breast_cancer.html', features=features, values=values, result=result)

if __name__ == '__main__':
    app.run(debug=True)
