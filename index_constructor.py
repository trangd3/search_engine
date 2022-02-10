from pymongo import MongoClient
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


# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore
# Use a service account
# cred = credentials.Certificate('serviceAccountKey.json')
# firebase_admin.initialize_app(cred)
# db = firestore.client()
# batch = db.batch()



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
        # print(id)
        # # testing cases
        # if index == 50:
        #     break;
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
    
    # adding to database
    # print("finished parsing")
    # for index, word in enumerate(unique_words):
    #     word_ref = db.collection("ex").document(word)
    #     batch.set(word_ref, {"id": unique_words[word]}, merge=True)

    #     # commits are limited to 500 operations
    #     if index > 0 and index % 9 == 0:
    #         print("committing")
    #         batch.commit()
    # print("final commit")
    # batch.commit()
                              
    return 0

index_constructor()
