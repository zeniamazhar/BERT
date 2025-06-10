# Fine-tuning BERT Model for Sentiment Analysis of IMDB Movie Reviews

## 1. Introduction
In this project, I aimed to use the IMDB movie reviews dataset, which consists of positive and negative reviews, to fine-tune a BERT model from Hugging Face to classify a given review as a positive or a negative one. The fine-tuned BERT model had an accuracy of 93.83%, a macro F1 score of 93.83%, and a balanced precision and recall for the negative and positive classes.

## 2. Problem Description
The IMDB movie reviews dataset consists of 2 columns: 
- The reviews
- The sentiment (positive or negative).

The review is the comment left by someone on a movie, while the sentiment reflects whether the review said positive things about the movie (positive review), or  negative things about it (negative review). This dataset contains a total of 50,000 reviews (25,000 positive and 25,000 negative reviews). The task in this project was to design a model aimed to perform binary sentiment analysis based on the reviews and predict whether a given review is positive or negative.

Fine-tuning BERT (Bidirectional Encoder Representations from Transformers) involves taking a pre-trained BERT model and adapting it to a specific task, such as text classification, by training it further on a labeled dataset. BERT is pre-trained on a large corpus using unsupervised objectives like masked language modeling and next sentence prediction, allowing it to learn rich contextual word embeddings—meaning each word’s representation is informed by the surrounding words in both directions. During fine-tuning, the model retains these embeddings and updates its internal weights using task-specific data, enabling it to capture nuances relevant to the new task while leveraging its deep understanding of language. This process typically involves adding a small task-specific layer on top of BERT (e.g., a classification head) and training the entire model end-to-end for a few epochs.

## 3. Methodology

### 3.1 Statistical analysis of the dataset
The dataset consists of 50,000 movie reviews, evenly split between positive and negative sentiments: 25,000 labeled as “positive” and 25,000 as “negative.” Each review is stored as a string of varying length. On average, reviews contain approximately 1,309 characters, with the shortest review being 32 characters and the longest reaching up to 13,704 characters. While fine-tuning the BERT model for this task, the review lengths were calculated by splitting the reviews by spaces. The average length was about 231 words, with the minimum being 4 words and the maximum being 2470. To see if there is a bias in the review length between the positive and negative sentiment classes, the mean, min and max review lengths were calculated. As can be seen in figure 1, the review lengths were quite similar on average (mean), but the maximum lengths are quite different, with the maximum length being 1522 for the negative reviews and 2470 for the positive ones. This balanced class distribution and diversity in review lengths make the dataset suitable for binary sentiment classification tasks.

![](https://github.com/zeniamazhar/BERT/blob/main/descriptiveStatistics)
##### Figure 1. Descriptive statistics of review lengths of reviews separated by positive and negative sentiment classes

### 3.2 Fine-tuning the BERT model

#### Preprocessing Steps
Firstly, missing data was removed by using the dropna() method on the dataframe obtained from the dataset, which would make it so there aren’t any missing values causing issues in the training or testing/validation stages. The sentiment labels were converted from categorical to numerical forms, by mapping the ‘positive’ sentiment class to 1, and the ‘negative’ sentiment class to 0, in order to fine-tune the BERT model with the data (since it’s not possible to do this with the categories directly). The BERTTokenizer was used to split the text into subwords units (i.e: unfriendly would get turned into “un”, “friend”, “ly”), and also truncate the reviews that are longer than 512 tokens, while padding the ones that are short (which allows shorter reviews to have the same shape as the longer ones).

#### Explanation of methods applied
The IMDB Dataset class was written in order to initialize the dataset with the tokenized inputs and their sentiment labels, to obtain the number of samples in the dataset, and to get one item in the dataset, which get converted to PyTorch tensors which allows them to be used to train the BERT model using Trainer and it also allows us to use DataLoader. 
The Trainer API from Hugging Face was used to train and evaluate the fine-tuned BERT model without manually writing the loops needed to train the model. The TrainingArguments class from Hugging Face’s transformers library allowed us to set the training arguments, including batch sizes, number of epochs, using L2 regularization (set by the line: weight_decay=0.01), saves the model after each epoch, and loads the best model at the end of training (based on F1 score). 
The compute_metrics function was written in order to calculate the evaluation metrics based on the predictions made on the validation set, including accuracy, precision, recall and macro F1 score (which is the average of the F1 scores obtained for both the positive and the negative classes).
The evaluate() method from the Trainer class allowed the fine-tuned BERT model to be evaluated on the test set, and return the loss, accuracy, f1 score, recall, and precision. Finally, the confusion matrix for this model was generated using the confusion_matrix and ConfusionMatrixDisplay from sklearn.metrics. The number of positive and negative class predictions and their true labels can be seen in the confusion matrix generated.

## 4. Results and Discussion

The fine-tuned BERT model performed really well on the test set, with a macro f1 score of 0.9383 and an accuracy of 93.83%. It can be seen in table 1 that the accuracy, precision, recall, and F1 scores were nearly identical (the rounded out versions seen in this table are all identical, but the raw numbers were slightly different). The fact that the precision and recall are similar shows that the number of true positives compared to the number of predicted positives (precision), and the number of true positives compared to the number of total positives (recall), were similar. This means that the model is predicting a similar number of false positives and false negatives, showing that the model has actually learned from the dataset. 

###### Table 1. Fine-tuned BERT model evaluation metrics on test set
| Metric     | Value   |
|------------|---------|
| Loss       | 25.7%   |
| Accuracy   | 93.83%  |
| Precision  | 93.83%  |
| Recall     | 93.83%  |
| F1 Score   | 93.83%  |

 
Moreover, the confusion matrix generated for the fine-tuned BERT model (figure 2) also shows that the number of false positives and false negatives were similar in number, with the number of false positives being 320, while the number of false negatives being 297. Majority of the
predictions made were correct, with 4641 correctly predicted negative reviews and 4742 correctly predicted positive reviews.

![](https://github.com/zeniamazhar/BERT/blob/main/confusionMatrix)
##### Figure 2. Confusion matrix generated for the fine-tuned BERT model

## 5. Conclusion
In this project, BERT model was fine-tuned in order to perform binary sentiment analysis on the IMDB movie reviews dataset. Among them, the fine-tuned BERT model performed really well on the test data, with a macro F1-score of 0.9383, and an accuracy of 0.9383. This was expected, since this is a deep-learning approach where contextual embeddings are used to understand the meaning of the words based on the words surrounding them (instead of just looking at each word separately), making it well-suited for this sentiment analysis task.

