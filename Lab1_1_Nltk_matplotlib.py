

from google.colab import drive
drive.mount("/content/drive")

# importing libraries
import nltk
import matplotlib.pyplot as plt
import pandas as pd

# Raw Text Analysis
random_text = """Discussing climate, sustainability, and preserving the natural world with President @EmmanuelMacron today in Paris. #BezosEarthFund #ClimatePledge"""

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

remove_link_text = re.sub(r'https?:\/\/.*[\r\n]*', '', random_text)
remove_link_text = re.sub(r'#', '', remove_link_text)
print(remove_link_text)

print('\033[92m' + random_text)
print('\033[92m' + remove_link_text)

from nltk.tokenize import sent_tokenize
text="""Hello Mr. steve, how you doing? whats up? The weather is great, and city is awesome. how you doing? The sky is pinkish-blue. You shouldn't eat cardboard, how you doing?"""
# download punkt
nltk.download("punkt")
tokenized_text=sent_tokenize(text)
print(tokenized_text)

# breaks paregraph into words
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

# frequency distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
fdist.most_common(4)
# Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30, cumulative = False, color = "green")
plt.show()

# stop words
from nltk.corpus import stopwords
# download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
print(stop_words)

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

# stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))

