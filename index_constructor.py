from asyncio import proactor_events
from concurrent.futures import process
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

tf = defaultdict(lambda: defaultdict(int))          # {docId: {lemma: term_count}}
df = defaultdict(int)                                 # {word: document_count}
words = defaultdict(dict)

def index_constructor():
    # {word: {docId: {"tf-idf": score, "in_metadata": boolean, "is_bolded": boolean, 
    #                 "in_title": boolean, "is_h1": boolean, "is_h2": boolean, "is_h3": boolean}}}


    # tf = defaultdict(dict)                              # {docId: {lemma: {score & booleans}}}
    # tf_idf = defaultdict(lambda: defaultdict(float))    # {word: {docId: score}}
    num_docs = 0

    bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))
    for index, (id, url) in enumerate(bk.items()):
        
        # testing cases
        print(id)
        # if index > 498:
        #     break;

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

                        # initializing fields for lemma in words dict
                        if lemma not in tf[id]:
                            words[lemma][id] = dict()
                            words[lemma][id]["metadata"] = False
                            words[lemma][id]["title"] = False
                            words[lemma][id]["bolded"] = False
                            words[lemma][id]["h1"] = False
                            words[lemma][id]["h2"] = False
                            words[lemma][id]["h3"] = False

                        # finding term frequency for each doc (how many times it appears in this document)
                        tf[id][lemma] += 1

                        # finding document frequency (how many documents the lemma appears in)
                        if lemma not in s:
                            df[lemma] += 1
                            s.add(lemma)

                # for calculating df (total number of valid documents)
                num_docs += 1


                # metadata
                metadata = soup.find("meta", attrs={"name":"description"})
                if metadata:
                    try:
                        metadata = metadata["content"]
                        process_text(metadata, "metadata")
                    except KeyError:
                        pass

                # title
                if soup.title:
                    title = soup.title.get_text().encode("ascii", "replace").decode().lower()
                    process_text(title, "title")

                # bold
                bolded = soup.find("b")
                if bolded:
                    bolded = bolded.get_text().encode("ascii", "replace").decode().lower()
                    process_text(bolded, "bolded")

                # h1
                h1 = soup.find("h1")
                if h1:
                    h1 = h1.get_text().encode("ascii", "replace").decode().lower()
                    process_text(h1, "h1")

                # h2
                h2 = soup.find("h2")
                if h2:
                    h2 = h2.get_text().encode("ascii", "replace").decode().lower()
                    process_text(h2, "h2")

                # h3
                h3 = soup.find("h3")
                if h3:
                    h3 = h3.get_text().encode("ascii", "replace").decode().lower()
                    process_text(h3, "h3")

                
    
    print("finished parsing")

    for id in tf:
        for lemma in tf[id]:
            words[lemma][id]["tfidf"] = (1 + math.log(tf[id][lemma], 10)) * math.log(num_docs/df[lemma])

    
    try:
        # ! https://pymongo.readthedocs.io/en/stable/examples/bulk.html
        db.test.bulk_write([InsertOne({"_id": term, "docId": words[term]}) for term in words])
        # db.example.bulk_write([InsertOne({"_id": word, "urls": unique_words[word]}) for word in unique_words])
    except BulkWriteError as bwe:
        print(bwe.details)
    
    print("finished writing")

    return 0

def process_text(text, tag):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for token in tokens:
        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
            lemma = lemmatizer.lemmatize(token)

            if lemma in tf[id]:
                words[lemma][id][tag] = True
