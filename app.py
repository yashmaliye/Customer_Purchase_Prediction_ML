import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained Gradient Boosting model
with open('Gradient.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature names based on model inspect metadata
# Features: ['age', 'gender', 'review', 'education']
# Note: Categorical features are encoded to numeric format for model input.

GENDER_MAP = {'Male': 0, 'Female': 1}
REVIEW_MAP = {'Poor': 0, 'Average': 1, 'Good': 2}
EDUCATION_MAP = {'School': 0, 'UG': 1, 'PG': 2}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    status_class = None
    
    # Retain input values for display in form
    form_data = {
        'age': '',
        'gender': 'Male',
        'review': 'Average',
        'education': 'UG'
    }

    if request.method == 'POST':
        try:
            # Extract form inputs (Categorical inputs in categorical string format)
            age = float(request.form.get('age', 0))
            gender_cat = request.form.get('gender', 'Male')
            review_cat = request.form.get('review', 'Average')
            education_cat = request.form.get('education', 'UG')

            # Update form_data dictionary to persist selections in UI
            form_data = {
                'age': age,
                'gender': gender_cat,
                'review': review_cat,
                'education': education_cat
            }

            # Map categorical string inputs to numeric encoding expected by model
            gender_num = GENDER_MAP.get(gender_cat, 0)
            review_num = REVIEW_MAP.get(review_cat, 1)
            education_num = EDUCATION_MAP.get(education_cat, 1)

            # Construct input array matching feature order: ['age', 'gender', 'review', 'education']
            input_features = np.array([[age, gender_num, review_num, education_num]])

            # Predict outcome
            prediction = model.predict(input_features)[0]

            if prediction == 1:
                prediction_text = "Status: PASS (Positive Output)"
                status_class = "pass"
            else:
                prediction_text = "Status: FAIL (Negative Output)"
                status_class = "fail"

        except Exception as e:
            prediction_text = f"Error processing input: {str(e)}"
            status_class = "fail"

    return render_template('index.html', 
                           prediction_text=prediction_text, 
                           status_class=status_class,
                           form_data=form_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
