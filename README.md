# SMS Spam Detection

A machine learning project that classifies SMS messages as spam or ham (not spam) using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

## Overview

This project demonstrates a complete text classification pipeline:

1. **Data Loading**: Loads the SMS Spam Collection dataset
2. **Text Preprocessing**: Lowercasing, punctuation removal, tokenization, stopword removal, and stemming
3. **Feature Extraction**: Uses CountVectorizer with unigrams and bigrams
4. **Model Training**: Multinomial Naive Bayes with hyperparameter tuning via GridSearchCV
5. **Model Persistence**: Saves the trained model using joblib

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd spam-detection

# Install dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd spam-detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas nltk scikit-learn jupyter jupyterlab notebook ipykernel
```

## Running the Notebook

### Using uv

```bash
# Start JupyterLab
uv run jupyter lab

# Or start Jupyter Notebook
uv run jupyter notebook
```

### Using pip (with virtual environment activated)

```bash
jupyter lab
# or
jupyter notebook
```

Then open `spamdetection.ipynb` in the Jupyter interface.

## Dataset

The project uses the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), which contains 5,572 SMS messages labeled as "spam" or "ham". The dataset file `SMSSpamCollection` should be in the project root directory.

## Project Structure

```
spam-detection/
├── SMSSpamCollection          # Dataset file
├── spamdetection.ipynb        # Main Jupyter notebook
├── spam_detection_model.joblib # Trained model (generated after running notebook)
├── pyproject.toml             # Project dependencies
├── uv.lock                    # Lock file for uv
└── README.md                  # This file
```

## Usage

After running the notebook, you can use the saved model to classify new messages:

```python
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the model
model = joblib.load('spam_detection_model.joblib')

# Preprocessing function
def preprocess_message(message):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Classify a message
message = "Congratulations! You've won a free iPhone!"
processed = preprocess_message(message)
prediction = model.predict([processed])
print("Spam" if prediction[0] == 1 else "Not Spam")
```

## Model Performance

The model uses GridSearchCV with 5-fold cross-validation to find the optimal alpha parameter for the Multinomial Naive Bayes classifier. The best model achieved optimal performance with `alpha=0.25`.

## License

This project is open source and available under the MIT License.
