# SMS Spam Detection Using Naive Bayes

## Project Overview
This project focuses on building a machine learning model to classify SMS messages as either **Ham** (legitimate) or **Spam**[cite: 3]. Developed as part of the **Introduction to Data Science Lab** at **Bahria University**, the system utilizes natural language processing (NLP) techniques and the Naive Bayes algorithm to achieve high classification accuracy.

## Dataset Information
The project uses the **SMSSpamCollection** dataset, which consists of 5,572 rows and 2 columns:
*   **label**: The target variable (ham or spam).
*   **message**: The raw text content of the SMS.

## Tech Stack
*   **Language**: Python.
*   **Libraries**:
    *   **Data Handling**: `pandas`, `NumPy`
    *   **NLP/Text Processing**: `nltk`, `string`, `re
    *   **Machine Learning**: `scikit-learn`
    *   **Visualization**: `seaborn`, `matplotlib`

## Data Preprocessing Steps
The project follows a rigorous text cleaning pipeline to prepare raw messages for the model:
1.  **Lowercasing**: Converting all text to lowercase to maintain uniformity.
2.  **Punctuation Removal**: Stripping special characters using Python’s `string.punctuation.
3.  **Numeral Removal**: Using regular expressions (`re.sub`) to remove numeric digits from the text.
4.  **Tokenization**: Splitting sentences into individual words (tokens) using `word_tokenize`.
5.  **Stop Word Removal**: Removing common English stop words (e.g., "is", "the", "in") that do not contribute to classification meaning.

## Implementation Details
*   **Feature Extraction**: The project utilizes `TfidfVectorizer` to convert cleaned text into numerical features.
*   **Model**: A Naive Bayes classifier is implemented to handle the probabilistic classification of text data.
*   **Validation**: The dataset is split into training and testing sets using `train_test_split`.

## Evaluation Metrics
The model's performance is measured using the following standard metrics:
*   Accuracy Score
*   Precision
*   Recall
*   F1-Score
*   Confusion Matrix (visualized via Seaborn)
