import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into numerical features using TF-IDF
from sklearn.svm import SVC  # Support Vector Classifier for classification
from sklearn.pipeline import Pipeline  # To chain transformers and estimators in a single object
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets
from sklearn.metrics import accuracy_score  # For evaluating the model's performance
import joblib  # For saving and loading the trained model

def train_model(data_path: str, model_path: str):
    """
    Function to train a machine learning model on a dataset and save the trained model.
    
    Args:
        data_path (str): Path to the dataset file (CSV format).
        model_path (str): Path to save the trained model.
    """
    # Load dataset from the given CSV file
    # The dataset should contain at least two columns: 'text' for input data and 'label' for target labels.
    data = pd.read_csv(data_path)  # Corrected line with quotes
    X, y = data['text'], data['label']  # Extract features (text) and labels
    
    # Split the dataset into training and testing sets
    # 80% of data is used for training, and 20% is used for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline that combines TF-IDF vectorization and Support Vector Machine classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Converts text data into TF-IDF feature matrix
        ('svm', SVC(kernel='linear', probability=True))  # SVM classifier with a linear kernel
    ])
    
    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Test the model by predicting labels on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate and print the accuracy of the model
    # Accuracy = (Number of Correct Predictions) / (Total Predictions)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Save the trained model to the specified file path
    # This allows for reusing the model without retraining
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")
# Entry point of the script
if __name__ == "__main__":
    # Specify the dataset file and output model path
    # Replace "dataset.csv" and "model/svm_model.pkl" with actual file paths as needed
    train_model("/Users/vamsinaidupirati/Desktop/MLOps/Text_classification/dataset.csv", "/Users/vamsinaidupirati/Desktop/MLOps/Text_classification/model/svm_model.pkl")
