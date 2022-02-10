from enum import unique
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
import json
# import lxml
# import math
import nltk
import os
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import RegexpTokenizer

client = MongoClient(port=27017)
db = client.search_engine

CORPUS_PATH = "../WEBPAGES_RAW"
STOPWORDS = set(stopwords.words('english'))
tm = defaultdict(lambda: wordnet.NOUN)
tm['J'] = wordnet.ADJ
tm['V'] = wordnet.VERB
tm['R'] = wordnet.ADV


def index_constructor():
    unique_words = defaultdict(lambda: defaultdict(float))
    
    bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))
    for index, (id, url) in enumerate(bk.items()):
        print(id)
        # testing cases
        if index == 500:
            break;
        with open(f"{CORPUS_PATH}/{id}", "r", encoding="utf-8") as file:
            text = file.read()

            # ! https://stackoverflow.com/questions/56887086/validating-if-a-string-is-a-valid-html-in-python
            if bool(BeautifulSoup(text, "lxml").find()):
                t = BeautifulSoup(text, "lxml")
                t = t.get_text().encode("ascii", "replace").decode().lower()
                tokens = word_tokenize(t)
                lemmatizer = WordNetLemmatizer()

                # ! https://www.guru99.com/stemming-lemmatization-python-nltk.html
                for token, tag in pos_tag(tokens):
                    # * if token is not a stop word and is alphanumeric
                    if len(token) >= 3 and token not in STOPWORDS and token.isalnum():
                        lemma = lemmatizer.lemmatize(token, tm[tag[0]])
                        if len(lemma) >= 3:
                            unique_words[lemma][id] = index

    
    print("finished parsing")
    try:
        # ! https://pymongo.readthedocs.io/en/stable/examples/bulk.html
        db.example.bulk_write([InsertOne({"_id": word, "urls": unique_words[word]}) for word in unique_words])
    except BulkWriteError as bwe:
        print(bwe.details)
                              
    return 0

index_constructor()
