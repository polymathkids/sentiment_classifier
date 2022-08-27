# models.py
import nltk
import numpy as np
import random
import math
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from collections import Counter

from sentiment_data import *
from utils import *

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    Yes! As UnigramFeatureExtractor(), BigramFeatureExtractor(), and BetterFeatureExtractor()
    all inherit the FeatureExtractor(), you want to define the extract_feature() method accordingly
    under each to give you the count of occurrences (the bag-of-words) based on your grouping of words.
     What you are returning is a counter without having to consider the weights and initial values.

Also you should consider lower casing and punctuation removal along the way.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        #self.features = []

    def get_indexer(self): #subclass from Feature Extractor- defined here for subclass call
        return self.indexer #return Indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter: #from Feature Extractor- defined here for subclass call
        #give the count of occurrences (the bag-of-words) based on your grouping of words (from Piazza)

        # Farm the tokens from the given list of words in the sentence
        features = [] #initialize place to store the indexed token features
        for token in sentence:
            if token.isalpha(): # Retain alphabetic words
                lower_token = token.lower() # Convert the tokens into lowercase
                #use utils.py function to get index. if add-To_indexer is FALSE, then in test mode and index lookup is used
                #if TRUE-  then it will add it if it doesn't already exist or return the existing index
                token_index = self.indexer.add_and_get_index(lower_token, add_to_indexer)
                features.append(token_index)


        # Create a Counter with the lowercase tokens: bow_simple
        # will return a count of the occurrence of features in the sentence passed
        bow_simple = Counter(features)
        #self.features = features #un-Counterized array

        return bow_simple



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self): #subclass from Feature Extractor- defined here for subclass call
        return self.indexer #return Indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter: #from Feature Extractor- defined here for subclass call
        #give the count of occurrences (the bag-of-words) based on your grouping of words (from Piazza)

        # Farm the tokens from the given list of words in the sentence
        features = [] #initialize place to store the indexed token features
        end_index = len(sentence) - 1
        magic = "|" #seperator to make a magic word so bigrams can use existing indexer utility
        for word_index in range(end_index): #create 'magic words' bigrams and use inexer utility
            featureA = sentence[word_index].lower()
            featureB = sentence[word_index + 1].lower()
            bigram_token = featureA + magic + featureB
            #use utils.py function to get index. if add-To_indexer is FALSE, then in test mode
            # it will lookup token and return the existing index
            #if TRUE-  then it will add it if it doesn't already exist or return the existing index
            bigram_index = self.indexer.add_and_get_index(bigram_token, add_to_indexer)
            features.append(bigram_index)

        # Create a Counter with the lowercase tokens: bow_bigram
        # will return a count of the occurrence of features in the sentence passed
        bow_bigram = Counter(features)
        return bow_bigram

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """


    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):  # subclass from Feature Extractor- defined here for subclass call
        return self.indexer  # return Indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.

        This features extractor removes all stop words and lemmatizes the words with the idea that it will
        generate fewer tokens but have higher overall predictive tokens as a result
        """

        def extract_features(self, sentence: List[str],
                             add_to_indexer: bool = False) -> Counter:  # from Feature Extractor- defined here for subclass call
            # give the count of occurrences (the bag-of-words) based on your grouping of words (from Piazza)
            # Farm the tokens from the given list of words in the sentence
            features = []  # initialize place to store the indexed token features
            # initialize list of all stop words
            english_stopwords = stopwords.words('english')
            # Instantiate the WordNetLemmatizer: wordnet_lemmatizer
            wordnet_lemmatizer = WordNetLemmatizer()

            for token in sentence:
                if token.isalpha():  # Retain alphabetic words, removes punctuation
                    if token not in english_stopwords: #remove stopwords, skip if a stop_word
                        lower_token = token.lower()  # Convert the tokens into lowercase
                        lemmatized = wordnet_lemmatizer.lemmatize(lower_token) #pares words to base/singular form (dogs -> dog)
                        # use utils.py function to get index. if add-To_indexer is FALSE, then in test mode and index lookup is used
                        # if TRUE-  then it will add it if it doesn't already exist or return the existing index
                        token_index = self.indexer.add_and_get_index(lemmatized, add_to_indexer)
                        features.append(token_index)

            # Create a Counter with the lowercase tokens: bow_better
            # will return a count of the occurrence of features in the sentence passed
            bow_better = Counter(features)

            return bow_better



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor):
        #init feature extractor
        self.feature_extractor = feature_extractor
        self.indexer = feature_extractor.get_indexer
        #use the indexer function to grab size of indexed features vector
        self.features_size = self.indexer.__len__()
        #initialize a weights vector -make it the size of indexed features vector
        self.weights_vector = np.zeros((self.features_size,), dtype = int) #np.zeros((5,), dtype=int) from help NOTE: maybe try evaluate with float
        #self.bow_array = np.zeros(self.weights_vector.shape) #initialize an array to un-Counterize the bow to match up to the weights

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        #get dictionary from the Counter of the indexed tokens
        # bow comes back as token_index, count for key ,value
        # set add_to_indexer to FALSE (predict run at testing) so indexer size is not changed
        bow_dict = self.feature_extractor.extract_features(sentence, add_to_indexer=False)



        def __init__(self, feature_extractor): #del
            self.feature_extractor = feature_extractor #del

            self.indexer = self.feature_extractor.get_indexer() #del
            self.bag_size = self.indexer.__len__() #del

            self.w = np.zeros((self.bag_size,), dtype=int) #del

        def featurizer(self, sentence: List[str]): #del
            return self.feature_extractor.extract_features(sentence) #del

        def predict(self, sentence: List[str]) -> int: #del
            return y_pred

        def update_w(self, y, y_pred, stepsize, sentence: List[str]): #del


    def prob(exponent):
        # print(1 / (1 + math.exp(-exponent))) #del
        return 1 / (1 + math.exp(-exponent)) #del


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    np.random.shuffle(array) (Throughout this course, the examples in our training sets not necessarily
    randomly ordered. You should make sure to randomly shuffle the data before iterating through it.
    Even better, you could do a random shuffle every epoch.)
    """
    alpha = 1  #step_size
    num_epochs = 3
    random.seed(42) #the answer to life and everything

    raise Exception("Must be implemented")


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model