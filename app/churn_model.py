import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
def load_data(file_path='/Users/bhuvan/churn_model/data/churn_data.csv'):
    """Load the dataset from the given file path."""
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure the file exists.")

# Preprocess data
def preprocess_data(data):
    """
    Preprocess the dataset:
    - Handle missing values.
    - Convert categorical variables to dummy variables.
    - Split into features (X) and target (y).
    """
    print("Preprocessing data...")
    data = data.dropna()  # Drop rows with missing values
    data = pd.get_dummies(data, drop_first=True)  # Convert categorical features to dummies
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    print(f"Data preprocessing complete. Features: {X.shape[1]}, Rows: {X.shape[0]}.")
    return X, y

# Train model
def train_model(X, y, model_path='model/churn_model.pkl', scaler_path='model/scaler.pkl'):
    """
    Train a logistic regression model and save it along with the scaler.
    - Splits the data into training and testing sets.
    - Standardizes the features.
    - Saves the trained model and scaler to disk.
    """
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Ensure the model directory exists
    os.makedirs('model', exist_ok=True)

    # Save the model
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to {model_path}.")

    # Save the scaler
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_path}.")

    return model, scaler

# Main function
def main():
    """
    Main function to load data, preprocess it, train the model, and save it.
    """
    try:
        data = load_data()
        X, y = preprocess_data(data)
        train_model(X, y)
        print("Model training and saving complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
