# sentiment_classifier
basic sentiment classifier with linear regression from scratch in python

## Assignment 1: Sentiment Classification

## Dataset and Code

Data:

You’ll be using the movie review dataset of Socher et al. (2013). This is a dataset of movie re- view snippets taken from Rotten Tomatoes. The labeled data actually consists of full parse trees with each constituent phrase of a sentence labeled with sentiment (including the whole sentence). The labels are “fine-grained” sentiment labels ranging from 0 to 4: highly negative, negative, neutral, positive, and highly positive.
We are tackling a simplified version of this task which frequently appears in the literature: positive/negative binary sentiment classification of sentences, with neutral sentences discarded from the dataset. The data files given to you contain of newline-separated sentiment examples, consisting of a label (0 or 1) followed by a tab, followed by the sentence, which has been tokenized but not lowercased. The data has been split into a train, development (dev), and blind test set. On the blind test set, you do not see the labels and only the sentences are given to you. The framework code reads these in for you.
Getting started Download the code and data. Expand the file and change into the directory.
To confirm everything is working properly, run:

python sentiment_classifier.py --model TRIVIAL --no_run_on_test

This loads the data, instantiates a TrivialSentimentClassifier that always returns 1 (positive), and evaluates it on the training and dev sets. The reported dev accuracy should be Accuracy: 444 / 872 = 0.509174.Alwayspredictingpositiveisn’tsogood!

## Framework code

The framework code you are given consists of several files. sentiment classifier.py is the main class.
The main method loads in the data, initializes the feature extractor, trains the model, and evaluates it on train, dev, and blind test, and writes the blind test results to a file.
Data reading is handled in sentiment data.py. This also defines a SentimentExample object, which wraps a list of words with an integer label (0/1).
utils.py implements an Indexer class, which can be used to maintain a bijective mapping between indices and features (strings).
models.py is the primary file that is modified
It defines base classes for the FeatureExtractor and the classifiers, and defines train perceptron and train logistic regression methods,
which you will be implementing. train model is your entry point which you may modify if needed.

### Part 1: Perceptron (40 points)

Q1 (40 points) Implement unigram perceptron. To receive full credit on this part, you must get at least 74% accuracy on the development set, and the training and evaluation (the printed time) should take less than20seconds.2 Notethatit’sfinetouseyourlearningrateschedulesfromQ2toachievethisperformance.

### Part 2: Logistic Regression (30 points)

Q2 (30 points) Implement logistic regression. Report your model’s performance on the dataset. You must get at least 77% accuracy on the development set and it must run in less than 20 seconds.

### Part 3: Features (30 points)

In this part, you’ll be implementing a more sophisticated set of features. You should implement two addi- tional feature extractors BigramFeatureExtractor and BetterFeatureExtractor. Note that your features for this can go beyond word n-grams; for example, you could define a FirstWord=X to extract a feature based on what first word of a sentence is, although this one may not be useful.

Q3 (15 points) Implement and experiment with BigramFeatureExtractor. Bigram features should be indicators on adjacent pairs of words in the text.
Q4 (15 points) Experiment with at least one feature modification in BetterFeatureExtractor. Try it out with logistic regression. Things you might try: other types of n-grams, tf-idf weighting, clipping your word frequencies, discarding rare words, discarding stopwords, etc. Your final code here should be whatever works best (even if that’s one of your other feature extractors). This model should train and evaluate in at most 60 seconds. This feature modification should not just consist of combining unigrams and bigrams.

## Command examples

python sentiment classifier.py --model PERCEPTRON --feats UNIGRAM python sentiment classifier.py --model LR --feats UNIGRAM

python sentiment classifier.py --model LR --feats BIGRAM

python sentiment classifier.py --model LR --feats BETTER

### References

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).
