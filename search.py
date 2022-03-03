import math
import numpy as np
from collections import defaultdict
from pymongo import MongoClient
import json

CORPUS_PATH = "../WEBPAGES_RAW"


class SearchEngine:
    def __init__(self):
        self.client = MongoClient(port=27017)
        self.db = self.client.search_engine
        self.collection = self.db.words
        self.bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))
        self.words = dict()

    def search(self):
        while True:
            # reset dictionary
            self.words = dict()
            search = input(
                "Search (Type 'quit!' to exit out): ").lower().strip()

            if search == "quit!":
                break

            query = list(set(search.split()))
            # don't lemmatize for now

            # calculations for query
            for word in query:
                self.words[word] = self.collection.find_one(word)

            query_frequencies = defaultdict(int)
            query_tfidf = defaultdict(float)
            query_norms = []

            for word in query:
                query_frequencies[word] += 1

            for word in query:
                if self.words[word]:
                    query_tfidf[word] = (
                        1 + math.log(query_frequencies[word], 10)) * self.words[word]["idf"]

            sum = 0
            for word in query:
                sum += math.pow(query_tfidf[word], 2)

            query_length = math.sqrt(sum)

            for word in query:
                query_norms.append(query_tfidf[word] / query_length)

            # calculations for documents
            doc_scores = defaultdict(float)
            documents = set()

            for word in query:
                for docId in self.words[word]["docId"]:
                    documents.add(docId)

            for doc in documents:
                doc_norms = []
                sum = 0
                for word in query:
                    if doc in self.words[word]["docId"]:
                        sum += math.pow(self.words[word]
                                        ["docId"][doc]["tfidf"], 2)

                doc_length = math.sqrt(sum)

                for word in query:
                    # print(x["docId"][doc]["tfidf"], doc_length )
                    try:
                        multiplier = 1
                        if self.words[word]["docId"][doc]["title"]:
                            multiplier += 0.3
                        if self.words[word]["docId"][doc]["metadata"]:
                            multiplier += 0.25
                        if self.words[word]["docId"][doc]["h1"]:
                            multiplier += 0.2
                        if self.words[word]["docId"][doc]["h2"]:
                            multiplier += 0.15
                        if self.words[word]["docId"][doc]["h3"]:
                            multiplier += 0.1
                        if self.words[word]["docId"][doc]["bolded"]:
                            multiplier += 0.05
                        norm = self.words[word]["docId"][doc]["tfidf"] * \
                            multiplier / doc_length
                        doc_norms.append(norm)
                    except KeyError:
                        doc_norms.append(0)

                doc_scores[doc] = float(np.dot(query_norms, doc_norms))

            for key, value in sorted(doc_scores.items(), key=lambda kv: -kv[1])[:20]:
                print(self.bk[key])
            print()
