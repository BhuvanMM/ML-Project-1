from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64

# Use the Agg backend for Matplotlib
plt.switch_backend('Agg')

# Initialize the Flask app
app = Flask(__name__)

# Load the model and scaler
MODEL_PATH = 'model/churn_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError("Model or scaler file not found. Ensure the trained model and scaler are saved in the 'model' directory.")

# Helper function to create graphs
def create_graph(features, values, title):
    """
    Generates a bar graph of the features and values.
    """
    plt.figure(figsize=(8, 6))
    plt.barh(features, values, color='skyblue')
    plt.xlabel("Feature Importance (Scaled Values)")
    plt.ylabel("Features")
    plt.title(title)
    plt.tight_layout()
    
    # Save graph to a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{graph_url}"

# Prediction function
def predict_churn(input_data):
    """
    Predicts churn based on input data.
    - Scales the input data using the saved scaler.
    - Uses the trained model to make a prediction.
    """
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)  # Scale the input
    prediction = model.predict(input_scaled)
    return prediction[0], input_scaled[0]

# Home route
@app.route('/')
def index():
    """
    Renders the homepage with the form for entering customer data.
    """
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submissions, validates input data, and returns prediction results.
    """
    try:
        # Collect input data from the form
        input_data = [float(x) for x in request.form.values()]
        
        # Check if the number of features matches the expected count
        expected_features = 13
        if len(input_data) != expected_features:
            error_message = f"Expected {expected_features} features, but got {len(input_data)}."
            return render_template('results.html', result=error_message, insights=None)

        # Predict churn
        prediction, scaled_data = predict_churn(input_data)

        # Prepare result
        result = 'Churn' if prediction == 1 else 'No Churn'

        # Generate insights and graph (only if Churn)
        graph_url = None
        insights = None
        if result == 'Churn':
            feature_names = [
                "Credit Score", "Geography", "Gender", "Age", "Tenure",
                "Balance", "Number of Products", "Has Credit Card",
                "Is Active Member", "Estimated Salary", 
                "Customer Satisfaction Score", "Total Transactions", "Has Phone Service"
            ]
            feature_importance = dict(zip(feature_names, scaled_data))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
            insights = sorted_features[:3]  # Select top 3 lowest features
            
            # Generate graph
            features, values = zip(*sorted_features[:5])  # Show top 5 features in the graph
            graph_url = create_graph(features, values, "Top Contributing Features for Churn")

        return render_template('results.html', result=f"Prediction: {result}", insights=insights, graph_url=graph_url)

    except ValueError as ve:
        # Handle value errors from incorrect input
        error_message = f"Invalid input values. Please enter numeric values. Error: {ve}"
        return render_template('results.html', result=error_message, insights=None)

    except Exception as e:
        # Handle other errors
        error_message = f"An unexpected error occurred: {e}"
        return render_template('results.html', result=error_message, insights=None)


# Run the Flask app
if __name__ == '__main__':
    # Ensure model directory exists before running the app
    if not os.path.exists('model'):
        print("Error: 'model' directory not found. Please ensure the trained model and scaler are available.")
    else:
        app.run(debug=True)
