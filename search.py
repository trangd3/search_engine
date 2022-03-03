import json
import math
import numpy as np
from collections import defaultdict
from pymongo import MongoClient

CORPUS_PATH = "../WEBPAGES_RAW"
bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))


class SearchEngine:
    def __init__(self):
        self.client = MongoClient(port=27017)
        self.db = self.client.search_engine
        self.collection = self.db.words
        self.words = dict()


    def calculate_query_norm(self, query):
        # calculations for query
        query_frequencies = defaultdict(int)
        query_tfidf = defaultdict(float)
        query_norms = []

        # count frequencies of word in query
        for word in query:
            query_frequencies[word] += 1

        # calculate the tfidf of each word within the query
        for word in query:
            tf = math.log(query_frequencies[word], 10) + 1
            query_tfidf[word] = tf * self.words[word]["idf"]

        # calculate the query length for normalization
        sum = 0
        for word in query:
            sum += math.pow(query_tfidf[word], 2)
        query_length = math.sqrt(sum)

        # calculate the normalized weight of each word's score
        for word in query:
            if query_length != 0:
                query_norms.append(query_tfidf[word] / query_length)

        return query_norms


    def calculate_doc_scores(self, query, query_norms):
        # calculations for documents
        documents = set()
        doc_scores = defaultdict(float)

        for word in query:
            for docId in self.words[word]["docId"]:
                documents.add(docId)

        for doc in documents:
            doc_norms = []
            sum = 0
            for word in query:
                # if self.words[word]["docId"][doc]:
                try:
                    sum += math.pow(self.words[word]["docId"][doc]["tfidf"], 2)
                except KeyError:
                    pass

            doc_length = math.sqrt(sum)

            for word in query:
                # if self.words[word]["docId"][doc]:
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
                # else:
                except KeyError:
                    doc_norms.append(0)

            doc_scores[doc] = float(np.dot(query_norms, doc_norms))

        return doc_scores

    def print_results(self, doc_scores):
        for docID, _ in doc_scores:
            print(bk[docID])
        print()

    def search(self):
        while True:
            # reset dictionary for new query
            self.words.clear()

            search = input("Search (type 'quit!' to exit out): ").lower().strip()

            # break out of infinite loop if user wants to quit
            if search == "quit!":
                break

            # only include words that are in our database
            query = search.split()
            modified_query = []
            for word in query:
                self.words[word] = self.collection.find_one(word)
                # only check for words that return something from the database
                if self.words[word]:
                    if word == "k":
                        print(self.words[word])
                    modified_query.append(word)

            if len(modified_query) == 0: 
                print("Results not found\n")

            else:
                query_norms = self.calculate_query_norm(modified_query)
                doc_scores = self.calculate_doc_scores(modified_query, query_norms)
                top_20 = sorted(doc_scores.items(), key=lambda kv: -kv[1])[:20]
                self.print_results(top_20)
