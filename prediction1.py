from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)

# Custom CORS middleware function
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Adjust as needed
    response.headers['Access-Control-Allow-Methods'] = 'POST'  # Adjust as needed
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Adjust as needed
    return response

app.after_request(add_cors_headers)  # Apply the middleware to all routes

# Load your trained model from the .pkl file
model = joblib.load("XGBClassifier_model.pkl")

# Define a route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    data = request.get_json()

    # Make a prediction using your model
    age = float(data['age'])
    sex = float(data['sex_selection'])
    pain_type = float(data['pain_type'])
    resting_bp = float(data['resting_bp'])
    cholesterol = float(data['cholesterol'])
    fasting_bs = float(data['fasting_bs'])
    resting_ecg = float(data['resting_ecg'])
    max_hr = float(data['max_hr'])
    exercise_angina = float(data['exercise_angina'])
    oldpeak = float(data['oldpeak'])
    st_slope = float(data['st_slope'])

    prediction = model.predict([[age, sex, pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
    prediction_probabilities = model.predict_proba([[age, sex, pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    # Convert the prediction to a meaningful result (e.g., "Positive" or "Negative")
    result = "Positive" if prediction[0] == 1 else "Negative"
    positive_probability = prediction_probabilities[0][1] * 100

    # Return the prediction result as JSON
    return jsonify({'result': result, 'probability_positive': positive_probability})

if __name__ == '__main__':
    app.run(debug=True)
