# nltk.download()
import math
import numpy as np
from collections import defaultdict
from pymongo import MongoClient
import json

CORPUS_PATH = "../WEBPAGES_RAW"

if __name__ == "__main__":
    client = MongoClient(port=27017)
    db = client.search_engine
    collection = db.words

    bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))
    # x = collection.find_one(search)

    # print((k, v) for k, v in sorted(x["docId"].items(), key = lambda p: -p[1]["tfidf"]))

    # for k, v in sorted(x["docId"].items(), key = lambda p: -p[1]["tfidf"]):
    #     print(k, v)

    while True:
        search = input("Search (Type 'quit!' to exit out): ").lower()

        if search == "quit!":
            break

        query = list(set(search.split()))
        # don't lemmatize for now

        # calculations for query
        words = dict()
        for word in query:
            words[word] = collection.find_one(word)

        query_frequencies = defaultdict(int)
        query_tfidf = defaultdict(float)
        query_norms = []

        for word in query:
            query_frequencies[word] += 1

        for word in query:
            if words[word]:
                query_tfidf[word] = (
                    1 + math.log(query_frequencies[word], 10)) * words[word]["idf"]

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
            for docId in words[word]["docId"]:
                documents.add(docId)

        for doc in documents:
            doc_norms = []
            sum = 0
            for word in query:
                if doc in words[word]["docId"]:
                    sum += math.pow(words[word]["docId"][doc]["tfidf"], 2)

            doc_length = math.sqrt(sum)

            for word in query:
                # print(x["docId"][doc]["tfidf"], doc_length )
                try:
                    multiplier = 1
                    if words[word]["docId"][doc]["title"]:
                        multiplier += 0.3
                    if words[word]["docId"][doc]["metadata"]:
                        multiplier += 0.25
                    if words[word]["docId"][doc]["h1"]:
                        multiplier += 0.2
                    if words[word]["docId"][doc]["h2"]:
                        multiplier += 0.15
                    if words[word]["docId"][doc]["h3"]:
                        multiplier += 0.1
                    if words[word]["docId"][doc]["bolded"]:
                        multiplier += 0.05
                    norm = words[word]["docId"][doc]["tfidf"] * \
                        multiplier / doc_length
                    doc_norms.append(norm)
                except KeyError:
                    doc_norms.append(0)

            doc_scores[doc] = float(np.dot(query_norms, doc_norms))

        for key, value in sorted(doc_scores.items(), key=lambda kv: -kv[1])[:20]:
            print(bk[key])
        print()

    # print((k, v) for k, v in sorted(x["docId"].items(), key = lambda p: -p[1]["tfidf"]))

    # for k, v in sorted(x["docId"].items(), key = lambda p: -p[1]["tfidf"]):
    #     print(k, v)
