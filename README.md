
# Sentiment Analysis on IMDB Dataset

This project demonstrates a sentiment analysis workflow on the IMDB dataset. The primary goal is to classify movie reviews as positive or negative using Natural Language Processing (NLP) techniques and machine learning models.

## Dataset

The dataset used in this project is the IMDB dataset, which contains movie reviews along with their associated sentiment (positive/negative). The data is loaded from a CSV file named `IMDB Dataset.csv`. You can download it from kaggle

## Workflow

1. **Data Loading**: 
   - The data is loaded using pandas.

2. **Data Splitting**:
   - The dataset is split into training (80%) and testing (20%) subsets using `train_test_split` from sklearn.

3. **Text Preprocessing**:
   - Special characters are removed from the text.
   - The text is converted to lowercase.
   - Stopwords are removed using the NLTK library.
   - Lemmatization is performed to reduce words to their base forms.

4. **Feature Extraction**:
   - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to convert text data into numerical features. 
   - The vectorizer is limited to a maximum of 500 features, ignoring terms that have a document frequency lower than 100.

5. **Modeling**:
   - A Decision Tree Classifier is used for the classification task.
   - Hyperparameter tuning is performed using GridSearchCV to find the best model configuration.

6. **Model Evaluation**:
   - The model's performance is evaluated using a confusion matrix and F1 score for both the training and testing datasets.

## Files

- `IMDB Dataset.csv`: The dataset containing movie reviews and their corresponding sentiment labels.
- `main.py`: The main script containing all the steps from loading the data to evaluating the model.

## Prerequisites

To run this project, you need to have the following packages installed:

- pandas
- scikit-learn
- nltk
- regex

You can install the required packages using the following command:

```bash
pip install pandas scikit-learn nltk regex
```

## How to Run

1. Ensure that the `IMDB Dataset.csv` file is in the same directory as the script.
2. Run the `main.py` script to execute the entire workflow.

```bash
python main.py
```

## Results

The final output of the model includes the confusion matrices and F1 scores for both the training and testing datasets, allowing you to assess the model's performance.

