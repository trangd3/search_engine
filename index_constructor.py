import json
from turtle import pos
import lxml
import math
import nltk
import os
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

STOPWORDS = set(stopwords.words('english'))
tm = defaultdict(lambda: wordnet.NOUN)
tm['J'] = wordnet.ADJ
tm['V'] = wordnet.VERB
tm['R'] = wordnet.ADV


def index_constructor():
    # https://www.tutorialspoint.com/python/os_walk.htm
    for root, dirs, files in os.walk("./0", topdown=True):
        for file in files:
            p = os.path.join(root, file)
            with open(p, "r", encoding = "utf-8") as f:
                text = f.read()
                # https://stackoverflow.com/questions/56887086/validating-if-a-string-is-a-valid-html-in-python
                if bool(BeautifulSoup(text, "lxml").find()):
                    t = BeautifulSoup(text, "lxml")
                    t = t.get_text().lower()
                    tokens = word_tokenize(t)
                    lemmatizer = WordNetLemmatizer()

                    # https://rustyonrampage.github.io/text-mining/2017/11/06/tokenization-with-python-and-nltk.html
                    # if [\w']+ is a weird tokenizer delimiter, use [a-zA-Z']+|\d+
                    # tokenizer = RegexpTokenizer("[\w']+")
                    # tokens = tokenizer.tokenize(t)
                    
                    # https://www.guru99.com/stemming-lemmatization-python-nltk.html
                    for token, tag in pos_tag(tokens):
                        # if token is not a stop word and is alphanumeric
                        if len(token) >= 3 and token not in STOPWORDS and token.isalnum():
                            lemma = lemmatizer.lemmatize(token, tm[tag[0]])
    return 0

index_constructor()

