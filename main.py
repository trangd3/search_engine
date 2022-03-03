# nltk.download()
import math
import numpy as np
from collections import defaultdict
from pymongo import MongoClient
import json
from search import SearchEngine

CORPUS_PATH = "../WEBPAGES_RAW"

if __name__ == "__main__":
    s = SearchEngine()
    s.search()
