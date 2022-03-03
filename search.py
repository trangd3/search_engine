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

    def calculate_query_norm(self, query):
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
            else:
                query_tfidf[word] = 0

        sum = 0
        for word in query:
            sum += math.pow(query_tfidf[word], 2)

        query_length = math.sqrt(sum)

        for word in query:
            if query_length != 0:
                query_norms.append(query_tfidf[word] / query_length)

        return query_norms

    def calculate_doc_scores(self, query, query_norms):
        # calculations for documents
        documents = set()
        doc_scores = defaultdict(float)

        for word in query:
            # if word in self.words:
            if self.words[word]:
                for docId in self.words[word]["docId"]:
                    documents.add(docId)

        for doc in documents:
            doc_norms = []
            sum = 0
            for word in query:
                if self.words[word] and self.words[word]["docId"][doc]:
                    sum += math.pow(self.words[word]["docId"][doc]["tfidf"], 2)

            doc_length = math.sqrt(sum)

            for word in query:
                # print(x["docId"][doc]["tfidf"], doc_length )
                if self.words[word] and self.words[word]["docId"][doc]:
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
                else:
                    doc_norms.append(0)

            doc_scores[doc] = float(np.dot(query_norms, doc_norms))

        return doc_scores

    def print_results(self, doc_scores):
        for docID, _ in doc_scores:
            print(self.bk[docID])
        print()

    def search(self):
        while True:
            # reset dictionary
            self.words = dict()
            search = input(
                "Search (Type 'quit!' to exit out): ").lower().strip()

            if search == "quit!":
                break

            query = list(set(search.split()))
            query_norms = self.calculate_query_norm(query)

            if len(query_norms) != 0:
                doc_scores = self.calculate_doc_scores(query, query_norms)
                top_20 = sorted(doc_scores.items(), key=lambda kv: -kv[1])[:20]
                self.print_results(top_20)
            else:
                print("Results not found")
