from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError
import json
# import lxml
import math
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
    num_docs = 0
    tf = defaultdict(lambda: defaultdict(int))          # {docId: {lemma: term_count}}
    df = defaultdict(int)                               # {word: document_count}
    tf_idf = defaultdict(lambda: defaultdict(float))    # {word: {docId: score}}
    # docs = defaultdict(lambda: defaultdict(list))

    '''
    word: {word: {docId: : score}}
    docs: {docId: {"metadata": [], "title": [], "bold": [], h1: [], h2: [], h3: []}}
    '''

    bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))
    for index, (id, url) in enumerate(bk.items()):

        # testing cases
        print(id)
        if index > 498:
            break;

        with open(f"{CORPUS_PATH}/{id}", "r", encoding="utf-8") as file:
            content = file.read()

            # ! https://stackoverflow.com/questions/56887086/validating-if-a-string-is-a-valid-html-in-python
            if bool(BeautifulSoup(content, "lxml").find()):
                # soupify link and tokenize text
                soup = BeautifulSoup(content, "lxml")
                lemmatizer = WordNetLemmatizer()

                text = soup.get_text().encode("ascii", "replace").decode().lower()
                tokens = word_tokenize(text)
                s = set()

                # ! https://www.guru99.com/stemming-lemmatization-python-nltk.html
                for token, tag in pos_tag(tokens):
                    # * if token is at least 3 characters long, not a stop word, and is alphanumeric
                    if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                        # lemmatize according to the token's first tag
                        lemma = lemmatizer.lemmatize(token, tm[tag[0]])

                        # finding term frequency for each doc (how many times it appears in this document)
                        tf[id][lemma] += 1

                        # finding document frequency (how many documents the lemma appears in)
                        if lemma not in s:
                            df[lemma] += 1
                            s.add(lemma)

                # for calculating df (total number of valid documents)
                num_docs += 1

                
                # if (soup.title is not None):
                #     title = soup.title.get_text().encode("ascii", "replace").decode().lower()
                #     tokens = word_tokenize(title)
                #     lst = []
                #     for token in tokens:
                #         lemma = lemmatizer.lemmatize(token)
                #         lst.append(lemma)
    
    print("finished parsing")

    for id in tf:
        for lemma in tf[id]:
            tf_idf[lemma][id] = (1+math.log(tf[id][lemma],10)) * math.log(num_docs/df[lemma])
    
    try:
        # ! https://pymongo.readthedocs.io/en/stable/examples/bulk.html
        db.test.bulk_write([InsertOne({"_id": term, "urls": tf_idf[term]}) for term in tf_idf])
        # db.example.bulk_write([InsertOne({"_id": word, "urls": unique_words[word]}) for word in unique_words])
    except BulkWriteError as bwe:
        print(bwe.details)
    
    print("finished writing")

    return 0

index_constructor()
