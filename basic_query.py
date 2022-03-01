import json
from pymongo import MongoClient

CORPUS_PATH = "../WEBPAGES_RAW"
client = MongoClient(port=27017)
db = client.search_engine


def query(word):
    return db.example.find_one({"_id": word})


def get_num_links(word):
    return len(query(word)['urls'])


def get_20_urls(word):
    bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))

    for index, docid in enumerate(query(word)['urls']):
        if index == 20:
            break
        print(f'{index + 1}. {bk[docid]}')


def summary_query():
    terms = ['informatics', 'mondego', 'irvine']

    for term in terms:
        print('===================================================')
        print('===================================================')
        print('===================================================')
        print(f'{term}: {get_num_links(term)}')
        get_20_urls(term)


if __name__ == '__main__':
    summary_query()
