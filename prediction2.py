from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your trained model from the .pkl file
model = joblib.load("XGBClassifier_model.pkl")

# Define a route for rendering the prediction form
@app.route('/')
def render_prediction_form():
    return render_template('prediction_form.html')

# Define a route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    age = float(request.form.get('age'))
    sex = float(request.form.get('sex_selection'))
    pain_type = float(request.form.get('pain_type'))
    resting_bp = float(request.form.get('resting_bp'))
    cholesterol = float(request.form.get('cholesterol'))
    fasting_bs = float(request.form.get('fasting_bs'))
    resting_ecg = float(request.form.get('resting_ecg'))
    max_hr = float(request.form.get('max_hr'))
    exercise_angina = float(request.form.get('exercise_angina'))
    oldpeak = float(request.form.get('oldpeak'))
    st_slope = float(request.form.get('st_slope'))

    # Make a prediction using your model
    prediction = model.predict([[age, sex, pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    # Convert the prediction to a meaningful result (e.g., "Positive" or "Negative")
    result = "Positive" if prediction[0] == 1 else "Negative"

    # Pass the result to the template for rendering
    return render_template('prediction_form.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
