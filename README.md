# sentiment_classifier
basic sentiment classifier with linear regression from scratch in python

## Sentiment Classification

## Dataset and Code

Data:

Using the movie review dataset of Socher et al. (2013), this is a dataset of movie re- view snippets taken from Rotten Tomatoes. The labeled data consists of full parse trees with each constituent phrase of a sentence labeled with sentiment (including the whole sentence). The labels are “fine-grained” sentiment labels ranging from 0 to 4: highly negative, negative, neutral, positive, and highly positive.
Goal: positive/negative binary sentiment classification of sentences, with neutral sentences discarded from the dataset. The data files contain newline-separated sentiment examples, consisting of a label (0 or 1) followed by a tab, followed by the sentence, which has been tokenized but not lowercased. The data has been split into a train, development (dev), and blind test set.
To confirm everything is working properly, run:

python sentiment_classifier.py --model TRIVIAL --no_run_on_test

This loads the data, instantiates a TrivialSentimentClassifier that always returns 1 (positive), and evaluates it on the training and dev sets. The reported dev accuracy should be Accuracy: 444 / 872 = 0.509174.Alwayspredictingpositiveisn’tsogood!

## Framework code

The given framework code consists of several files. sentiment classifier.py is the main class.
The main method loads in the data, initializes the feature extractor, trains the model, and evaluates it on train, dev, and blind test, and writes the blind test results to a file.
Data reading is handled in sentiment data.py. This also defines a SentimentExample object, which wraps a list of words with an integer label (0/1).
utils.py implements an Indexer class, which can be used to maintain a bijective mapping between indices and features (strings).
models.py is the primary file that is modified
It defines base classes for the FeatureExtractor and the classifiers, and defines train perceptron and train logistic regression methods.

### Implementation:

Implement unigram perceptron (from scratch, no packages). Implement logistic regression (from scratch, no package help). Implement a more sophisticated set of feature extractors: BigramFeatureExtractor and BetterFeatureExtractor. Bigram features are indicators on adjacent pairs of words in the text. BetterFeatureExtractor experiemnts with different approaches such as logistic regression, other types of n-grams, tf-idf weighting, clipping word frequencies, discarding rare words, discarding stopwords, etc. The inal code is the result of what I found worked best from those experiments.

## Command examples

python sentiment classifier.py --model PERCEPTRON --feats UNIGRAM python sentiment classifier.py --model LR --feats UNIGRAM

python sentiment classifier.py --model LR --feats BIGRAM

python sentiment classifier.py --model LR --feats BETTER

### References

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).
