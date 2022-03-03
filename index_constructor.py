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

# {docId: {term: term_count}}
tf = defaultdict(lambda: defaultdict(int))
df = defaultdict(int)                               # {term: document_count}
idfs = defaultdict(float)                           # {term: idf}
# {term: {docId: {"tfidf": float, "metadata": bool, "bolded": bool,
words = defaultdict(dict)
#                 "title": bool, "h1": bool, "h2": bool, "h3": bool}}


def index_constructor():
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
                        term = lemmatizer.lemmatize(token, tm[tag[0]])

                        # initializing fields for term in words dict
                        if term not in tf[id]:
                            words[term][id] = dict()
                            words[term][id]["metadata"] = False
                            words[term][id]["title"] = False
                            words[term][id]["bolded"] = False
                            words[term][id]["h1"] = False
                            words[term][id]["h2"] = False
                            words[term][id]["h3"] = False

                        # finding term frequency for each doc (how many times it appears in this document)
                        tf[id][term] += 1

                        # finding document frequency (how many documents the term appears in)
                        if term not in s:
                            df[term] += 1
                            s.add(term)

                # for calculating df (total number of valid documents)
                num_docs += 1

                # metadata
                metadata = soup.find("meta", attrs={"name": "description"})
                if metadata:
                    try:
                        metadata = metadata["content"]
                        tokens = word_tokenize(metadata)
                        for token in tokens:
                            if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                                term = lemmatizer.lemmatize(token)

                                if term in tf[id]:
                                    words[term][id]["metadata"] = True
                    except KeyError:
                        pass

                # title
                if soup.title:
                    title = soup.title.get_text().encode("ascii", "replace").decode().lower()
                    tokens = word_tokenize(title)
                    for token in tokens:
                        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                            term = lemmatizer.lemmatize(token)

                            if term in tf[id]:
                                words[term][id]["title"] = True

                # bold
                bolded = soup.find("b")
                if bolded:
                    bolded = bolded.get_text().encode("ascii", "replace").decode().lower()
                    tokens = word_tokenize(bolded)
                    for token in tokens:
                        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                            term = lemmatizer.lemmatize(token)

                            if term in tf[id]:
                                words[term][id]["bolded"] = True

                # h1
                h1 = soup.find("h1")
                if h1:
                    h1 = h1.get_text().encode("ascii", "replace").decode().lower()
                    tokens = word_tokenize(h1)
                    for token in tokens:
                        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                            term = lemmatizer.lemmatize(token)

                            if term in tf[id]:
                                words[term][id]["h1"] = True

                # h2
                h2 = soup.find("h2")
                if h2:
                    h2 = h2.get_text().encode("ascii", "replace").decode().lower()
                    tokens = word_tokenize(h2)
                    for token in tokens:
                        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                            term = lemmatizer.lemmatize(token)

                            if term in tf[id]:
                                words[term][id]["h2"] = True

                # h3
                h3 = soup.find("h3")
                if h3:
                    h3 = h3.get_text().encode("ascii", "replace").decode().lower()
                    tokens = word_tokenize(h3)
                    for token in tokens:
                        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                            term = lemmatizer.lemmatize(token)

                            if term in tf[id]:
                                words[term][id]["h3"] = True

    print("finished parsing")

    for id in tf:
        for term in tf[id]:
            idfs[term] = math.log(num_docs/df[term], 10)
            words[term][id]["tfidf"] = (
                1 + math.log(tf[id][term], 10)) * idfs[term]

    try:
        # ! https://pymongo.readthedocs.io/en/stable/examples/bulk.html
        db.words.bulk_write([InsertOne(
            {"_id": term, "idf": idfs[term], "docId": words[term]}) for term in words])
        # db.example.bulk_write([InsertOne({"_id": term, "urls": unique_words[term]}) for term in unique_words])
    except BulkWriteError as bwe:
        print(bwe.details)

    print("finished writing")

    return 0

# def process_text(text, tag):
#     lemmatizer = WordNetLemmatizer()
#     tokens = word_tokenize(text)
#     for token in tokens:
#         if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
#             term = lemmatizer.lemmatize(token)

#             if term in tf[id]:
#                 words[term][id][tag] = True


index_constructor()
