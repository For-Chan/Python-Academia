# Modules and Libraries
import pandas as pd

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

''' Chardet library for detecting encoding of a file, analyzes a sample of file's contents.
Not 100% accurate with limited char sets or small files...
pip install chardet'''


# CONSTANTS
# List of a few encodings, can add to list for extensive file processing
ENCODINGS = ['utf-8', 'ISO-8859-1', 'cp1252', 'UTF-16', 'ASCII', 'Windows-1252']
# Store csv in variable
CSV_FILE = "globalWarming.csv"

# FUNCTIONS
''' Attempts to read csv file with various encoding types
Returns dataframe df after successful read'''
def read_csv_encoding(file_path, encodings=ENCODINGS):
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"File read successfully with encoding: {encoding}")
            return df
        except UnicodeDecodeError as e:
            print(f"Error reading file with encoding {encoding}: {e}")
    raise ValueError("All encodings failed. Please check file format.")

''' Preprocess the df
Returns DataFrame df after preprocess'''
def preprocess_data(df):
    # Validate the input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data is not a pandas DataFrame")
    
    # Rename columns
    df = df.rename(columns={'existence': 'Sentiment', 'tweet': 'Tweet'})
    
    # Remove rows with missing 'Sentiment' values
    df = df.dropna(subset=['Sentiment'])
    
    # Map 'Yes' and 'Y' to 1, 'No' and 'N' to 0
    df['Sentiment'] = df['Sentiment'].map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0})
    
    # Drop 'existence.confidence' column
    df = df.drop(columns=['existence.confidence'])
    
    return df

''' Split Dataframe df into training and testing set 
Function separates Dataframe into features and labels and splits them into training/testing sets '''
def split_data(df):
    # Split data into feature (Tweet) and labels (Sentiment)
    tweets = df['Tweet']
    sentiments = df['Sentiment']
    
    # Split data into training and testing sets
    # Training 40% and Testing sets 60%
    tweets_train, tweets_test, sentiments_train, sentiments_test = train_test_split(tweets, sentiments, test_size=0.6, random_state=42)
    
    return tweets_train, tweets_test, sentiments_train, sentiments_test

''' Converts text data to numerical data using TF-IDF Vectorization
Algorithms cannot process raw text directly.
Text data is inherently categorical and unstructured,
making it unsuitable for algorithms that require structured, numeric input
'''
def vectorize_text(tweets_train, tweets_test):
    # Verify input data is in expected format
    if not isinstance(tweets_train, pd.Series) or not isinstance(tweets_test, pd.Series):
        raise TypeError("Input data must be pandas Series.")
    
    try:
        # Intialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        
        # Fit vectorizer to the training data and tranform training data
        # Step learns the vocabulary of training data and converts it into TF-IDF vectors
        tweets_train_vect = vectorizer.fit_transform(tweets_train)
        
        # Transform testing data
        # Important to use the same vectorizer to maintain consistency between training and testing data
        tweets_test_vect = vectorizer.transform(tweets_test)
        
        # Get feature names (terms)
        feature_names = vectorizer.get_feature_names_out()
        
        return tweets_train_vect, tweets_test_vect, feature_names
    except Exception as e:
        print(f"An error occured during vectorization: {e}")
        # Handle the exception or re-raise it
        raise

''' Trains a K-Nearest Neighbor model and evaluates its performance
Function takes vectorized tweet data and sentiment labels for training,
and similar data for testing, to train a KNN model and evaluate its accuracy'''
def train_evaluate_knn(tweets_train_vect, sentiments_train, tweets_test_vect, sentiments_test):
    # Initialize KNN classifier
    # n_neighbors is a hyperparameter that you can tune. Starting with 5
    knn = KNeighborsClassifier(n_neighbors=5)
    
    try:
        # Train the KNN model using training data
        knn.fit(tweets_train_vect, sentiments_train)
        
        # Predict sentiments on the test dataset
        predictions = knn.predict(tweets_test_vect)
        
        # Evaluate model's performance
        accuracy = accuracy_score(sentiments_test, predictions)
        report = classification_report(sentiments_test, predictions)
        
        return accuracy, report
    except Exception as e:
        # Handle exceptions that may occur during training or evaluation
        print(f"An error occured during training/evaluating the KNN model: {e}")
        raise

''' MAIN FUNCTION to execute the workflow '''
def main():
    try:
        # Load data with error handling
        df = read_csv_encoding(CSV_FILE)
        print(f"Success: File processed into a DataFrame: {type(df)}")
        
        # Call to Preprocess data function
        df = preprocess_data(df)
        print("Success: DataFrame has been Preprocessed for analysis.")
        
        # Call to Split data into training and testing sets function
        tweets_train, tweets_test, sentiments_train, sentiments_test = split_data(df)
        print("Success: DataFrame has been split into training and testing sets")
        
        
        # Call to Vectorize text data into numerical data function
        tweets_train_vect, tweets_test_vect, feature_names = vectorize_text(tweets_train, tweets_test)
        print("Success: Vectorized text data into numerical data")
        # Convert a few Vectors to arrays and preview their content
        for i in range(3): # Preview first 3 documents
            print(f"Tweet {i}:")
            print("Feature Scores:", tweets_test_vect[i].toarray())
            print("Feature Names:", feature_names)
        
        # Call to Train and evaluate K-Nearest Neighbor model function
        accuracy, report = train_evaluate_knn(tweets_train_vect, sentiments_train, tweets_test_vect, sentiments_test)
        print("Success: Training and evaluation using KNN model")
        
        # Print results
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle or log exception


# CALL FOR MAIN FUNCTION
if __name__ == "__main__":
    main()
