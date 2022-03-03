# nltk.download()
from pymongo import MongoClient

if __name__ == "__main__":
    search = input("Search: ").lower()
    client = MongoClient(port=27017)
    db = client.search_engine
    collection = db.test
    x = collection.find_one(search)

    # print((k, v) for k, v in sorted(x["docId"].items(), key = lambda p: -p[1]["tfidf"]))

    for k, v in sorted(x["docId"].items(), key = lambda p: -p[1]["tfidf"]):
        print(k, v)