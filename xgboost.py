import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# File paths and parameters
TRAIN_CSV_PATH = '/kaggle/input/vscp-pml-unibuc-2024/train.csv'
VALIDATION_CSV_PATH = '/kaggle/input/vscp-pml-unibuc-2024/val.csv'
TEST_CSV_PATH = '/kaggle/input/vscp-pml-unibuc-2024/test.csv'

SEED = 42
EMBEDDING_DIM = 250

# XGBoost hyperparameters
XGBOOST_PARAMETERS = {
    'random_state': SEED,
    'n_estimators': 575,
    'learning_rate': 0.005,
    'max_depth': 8,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}

# Word2Vec hyperparameters
WORD2VEC_PARAMETERS = {
    'vector_size': EMBEDDING_DIM,
    'window': 5,
    'min_count': 2,
    'epochs': 50,
    'seed': SEED
}

class VisualSentenceComplexity:
    def __init__(self,
                 train_csv_path,
                 validation_csv_path,
                 test_csv_path,
                 word2vec_params,
                 xgboost_params,
                 embedding_dim):
        self.train_csv_path = train_csv_path
        self.validation_csv_path = validation_csv_path
        self.test_csv_path = test_csv_path
        self.word2vec_params = word2vec_params
        self.xgboost_params = xgboost_params
        self.embedding_dim = embedding_dim

        self.stop_words = None
        self.word2vec_model = None
        self.scaler = None
        self.xgboost_model = None
        self.idf_dictionary = None

    def get_stop_words(self):
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def load_datasets(self, train_csv, validation_csv, test_csv):
        # Read CSVs into DataFrames
        train_dataframe = pd.read_csv(train_csv)
        validation_dataframe = pd.read_csv(validation_csv)
        test_dataframe = pd.read_csv(test_csv)
        return train_dataframe, validation_dataframe, test_dataframe

    def preprocess_text(self, dataframe):
        # Apply text cleaning to the DataFrame
        dataframe['text'] = dataframe['text'].apply(self.preprocess_single_text)
        return dataframe

    def preprocess_single_text(self, text):
        # Lowercase, remove punctuation/numbers, split, and remove stop words
        text_lower = text.lower()
        text_no_punctuation = re.sub(r'[^\w\s]', '', text_lower)
        text_no_numbers = re.sub(r'\d+', '', text_no_punctuation)
        tokens = text_no_numbers.split()
        tokens_filtered = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens_filtered)

    def train_w2v(self, corpus):
        # Tokenize and train Word2Vec on the combined corpus
        tokenized_corpus = [document.split() for document in corpus]
        word2vec_model = Word2Vec(sentences=tokenized_corpus, **self.word2vec_params)
        return word2vec_model

    def fit_tfidf(self, corpus):
        # Fit a TF-IDF model to build an IDF dictionary
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(corpus)
        vocab = tfidf_vectorizer.vocabulary_
        idf_scores = tfidf_vectorizer.idf_
        self.idf_dictionary = {word: idf_scores[index] for word, index in vocab.items()}

    def get_wemb(self, tokens, word2vec_model, embedding_dim, idf_dictionary):
        # Get weighted embedding by IDF scores
        weighted_vectors = []
        for word in tokens:
            if word in word2vec_model.wv and word in idf_dictionary:
                weight = idf_dictionary[word]
                weighted_vectors.append(word2vec_model.wv[word] * weight)
        if not weighted_vectors:
            return np.zeros(embedding_dim)
        else:
            return np.mean(weighted_vectors, axis=0)

    def vectorize_texts(self, texts):
        return np.array([
            self.get_wemb(
                text.split(),
                self.word2vec_model,
                self.embedding_dim,
                self.idf_dictionary
            )
            for text in texts
        ])

    def fit_feature_scaler(self, feature_matrix):
        # Fit a StandardScaler and transform the features
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(feature_matrix)

    def transform_feature_scaler(self, feature_matrix):
        # Scale features with the already fitted scaler
        return self.scaler.transform(feature_matrix)

    def train_xgb(self, feature_matrix, target):
        # Train the XGBoost model
        self.xgboost_model = xgb.XGBRegressor(**self.xgboost_params)
        self.xgboost_model.fit(feature_matrix, target)

    def evaluate(self, true_values, predicted_values):
        # Evaluate model using Spearman correlation
        return spearmanr(true_values, predicted_values).correlation

    def execute_pipeline(self):
        self.get_stop_words()

        # Load training, validation, and test sets
        train_df, validation_df, test_df = self.load_datasets(
            self.train_csv_path,
            self.validation_csv_path,
            self.test_csv_path
        )

        # Clean the text in all sets
        train_df = self.preprocess_text(train_df)
        validation_df = self.preprocess_text(validation_df)
        test_df = self.preprocess_text(test_df)

         # Combine train and validation text for building Word2Vec and TF-IDF
        combined_corpus = pd.concat([train_df['text'], validation_df['text']])

        # Train Word2Vec
        self.word2vec_model = self.train_w2v(combined_corpus)

        # Build IDF dictionary
        self.fit_tfidf(combined_corpus)

        # Vectorize texts
        X_train = self.vectorize_texts(train_df['text'])
        y_train = train_df['score'].values

        X_validation = self.vectorize_texts(validation_df['text'])
        y_validation = validation_df['score'].values

        X_test = self.vectorize_texts(test_df['text'])
        test_ids = test_df['id'].values

        # Scale features
        X_train_scaled = self.fit_feature_scaler(X_train)
        X_validation_scaled = self.transform_feature_scaler(X_validation)
        X_test_scaled = self.transform_feature_scaler(X_test)

        # Train XGBoost
        self.train_xgb(X_train_scaled, y_train)

        # Evaluate on validation set
        validation_predictions = self.xgboost_model.predict(X_validation_scaled)
        validation_spearman = self.evaluate(y_validation, validation_predictions)
        print(f"Validation: {validation_spearman}")


        # Predict on test set and save submission
        test_predictions = self.xgboost_model.predict(X_test_scaled)
        submission = pd.DataFrame({'id': test_ids, 'score': test_predictions})
        submission.to_csv('submission.csv', index=False)

pipeline = VisualSentenceComplexity(
    TRAIN_CSV_PATH,
    VALIDATION_CSV_PATH,
    TEST_CSV_PATH,
    WORD2VEC_PARAMETERS,
    XGBOOST_PARAMETERS,
    EMBEDDING_DIM
)
pipeline.execute_pipeline()