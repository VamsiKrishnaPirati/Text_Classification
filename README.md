# Text Classification API with FastAPI

This project provides a text classification API built using FastAPI and an SVM model trained on text data. The API allows for predicting the sentiment or category of input text.

---

## Features

- **Endpoints:**
  - `/predict/`: Predict the sentiment or category of input text.
  - `/`: Welcome message for the API.
- **Interactive Documentation:**
  - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Pretrained SVM Model:** A pipeline including TF-IDF vectorization and an SVM classifier.
- **Monitoring:** Tracks prediction time for performance evaluation.

---

## Repository Structure

```plaintext
.
├── app
│   ├── main.py       # FastAPI app entry point
│   ├── routes.py     # API routes and logic
├── model
│   ├── train.py      # Script to train and save the model
│   ├── monitor.py    # Decorator for monitoring prediction time
├── dataset.csv        # Dataset used for training
├── requirements.txt   # Dependencies for the project
└── README.md          # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- `pip` for package installation

---

### Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv text_classification_env
   source text_classification_env/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Training the Model

1. Ensure `dataset.csv` is present in the root directory.
   - The dataset should contain:
     - `text`: Input text data
     - `label`: Target labels (e.g., sentiments or categories)

2. Run the training script:
   ```bash
   python model/train.py
   ```
   - The script trains the model and saves it as `model/svm_model.pkl`.
   - Training accuracy will be displayed in the terminal.

---

### Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
   - The API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

2. Access Interactive API Documentation:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Example Usage

1. **POST `/predict/`**
   - **Request:**
     ```json
     {
       "text": "I love this product, it's amazing!"
     }
     ```
   - **Response:**
     ```json
     {
       "status": "success",
       "data": {
         "prediction": "positive",
         "probability": [0.8, 0.2]
       }
     }
     ```

---

## Monitoring

The `monitor.py` script contains a decorator `@monitor_prediction_time` to log the prediction execution time for performance evaluation.

---

## File Descriptions

### `model/train.py`
- Trains the SVM model using TF-IDF features.
- Saves the trained model as `model/svm_model.pkl`.

### `model/monitor.py`
- Provides a decorator to monitor prediction times.

### `app/routes.py`
- Defines the `/predict/` endpoint to handle text classification.

### `app/main.py`
- Entry point for running the FastAPI application.
- Includes the router from `routes.py`.

---

## Dependencies

Install the following Python packages (specified in `requirements.txt`):

- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pandas`
- `joblib`

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'app'`**
  - Ensure you are in the correct directory.
  - Run the command from the project root directory.

- **`TypeError: monitor_prediction_time() missing 1 required positional argument`**
  - Ensure the `@monitor_prediction_time` decorator is applied correctly in `routes.py`.

---

## License

This project is licensed under the MIT License.

---

## Author

**Vamsi Krishna Pirati**
