
import nltk
from nltk.corpus import twitter_samples 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf

nltk.download('twitter_samples')
nltk.download('stopwords')
