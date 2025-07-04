SPAM & HAM – AN NLP PROJECT SUMMARY 

Objective:  

To build a deep learning–based binary classification model that accurately classifies SMS messages as spam or ham (not spam), using natural language processing (NLP) techniques. 

Dataset: Source: spam.csv  

Structure: Category: Target label (ham = 0, spam = 1) Message: Raw SMS text 

Step-by-Step Pipeline: 

Text Preprocessing Converted to lowercase Removed digits, punctuation, and extra whitespace Encoded Category to binary (0 for ham, 1 for spam) 

Class Imbalance Handling Dataset was imbalanced (~13% spam) Applied random oversampling only on the training set to balance spam and ham classes (avoiding data leakage) 

Train-Test Split Used train_test_split with stratification 80% training / 20% test split Balanced only the training portion to ensure real-world test evaluation 

NLP Architecture: 

Input (Padded Sequences)
   ↓
Embedding Layer (Word vectors)
   ↓
Bidirectional LSTM (64 units)
   ↓
GlobalMaxPooling1D
   ↓
Dropout (0.5)
   ↓
Dense Layer with Sigmoid Activation
   ↓
Output: Probability of Spam (1) or Ham (0)


Probability of Spam (1) or Ham (0)  

Embedding layer learns dense word representations  

Bidirectional LSTM captures context from both directions (forward + backward) 

 Global Max Pooling condenses the most relevant features Dropout prevents overfitting Sigmoid outputs probability score 

Training Details:  

Loss Function: Binary Cross entropy  

Optimizer: Adam  

Epochs: 15–20 (with Early Stopping)  

Validation Split: 20% of training set  

Batch Size: 32  

Evaluation Metrics: 

 Accuracy Precision, Recall, F1-score  

Confusion Matrix Achieved ~97% accuracy on test data  

Model was not overfitting due to:  

Balanced training data Early stopping Realistic test set Inference Pipeline Clean new message Tokenize + pad Pass into model Predict → Label as Spam or Ham 

Model Saving:  

Model saved as .h5 using model. Save () Tokenizer saved with pickle as .pkl Easily reloadable for inference or API integration 

Why LSTM for the SPAM& HAM? 

 Understands sequence and context Captures subtle spam patterns (e.g., “claim now” vs. “feel free to call”)  

Outperforms traditional ML models (like Naive Bayes or SVM) on short-text classification 

 
