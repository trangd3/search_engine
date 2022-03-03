import json
import math
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient, InsertOne
from pymongo.errors import BulkWriteError

CORPUS_PATH = "../WEBPAGES_RAW"
STOPWORDS = set(stopwords.words('english'))
bk = json.load(open(f"{CORPUS_PATH}/bookkeeping.json"))

class IndexConstructor:
    def __init__(self):
        self.client     = MongoClient(port=27017)
        self.db         = self.client.test

        self.tf         = defaultdict(lambda: defaultdict(int)) # {docId: {term: term_count}}
        self.df         = defaultdict(int)   # {term: document_count}
        self.idfs       = defaultdict(float) # {term: idf}
        self.words      = defaultdict(dict)  # {term: {docId: {tfidf: float, html_tags: bool}}}

        self.tm         = defaultdict(lambda: wordnet.NOUN)
        self.tm['J']    = wordnet.ADJ
        self.tm['V']    = wordnet.VERB
        self.tm['R']    = wordnet.ADV

        self.lemmatizer = WordNetLemmatizer()
        self.num_docs   = 0


    def construct_index(self):
        for index, (id, _) in enumerate(bk.items()):

            # testing cases
            print(id)
            if index > 49:
                break;

            with open(f"{CORPUS_PATH}/{id}", "r", encoding="utf-8") as file:
                content = file.read()

                # ! https://stackoverflow.com/questions/56887086/validating-if-a-string-is-a-valid-html-in-python
                if bool(BeautifulSoup(content, "lxml").find()):
                    # soupify link and tokenize text
                    soup = BeautifulSoup(content, "lxml")

                    text = soup.get_text().encode("ascii", "replace").decode().lower()
                    tokens = word_tokenize(text)
                    s = set()

                    # ! https://www.guru99.com/stemming-lemmatization-python-nltk.html
                    for token, tag in pos_tag(tokens):
                        # if token is at least 3 characters long, not a stop word, and is alphanumeric
                        if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                            # lemmatize according to the token's first tag
                            term = self.lemmatizer.lemmatize(token, self.tm[tag[0]])

                            # initializing fields for term in words dict
                            if term not in self.tf[id]:
                                self.words[term][id] = dict()
                                self.words[term][id]["metadata"] = False
                                self.words[term][id]["title"] = False
                                self.words[term][id]["bolded"] = False
                                self.words[term][id]["h1"] = False
                                self.words[term][id]["h2"] = False
                                self.words[term][id]["h3"] = False

                            # finding term frequency for each doc (how many times it appears in this document)
                            self.tf[id][term] += 1

                            # finding document frequency (how many documents the term appears in)
                            if term not in s:
                                self.df[term] += 1
                                s.add(term)

                    # for calculating df (total number of valid documents)
                    self.num_docs += 1

                    # metadata
                    metadata = soup.find("meta", attrs={"name": "description"})
                    if metadata:
                        # try/except because some pages might not have a content attribute
                        try:
                            metadata = metadata["content"]
                            self._process_data(metadata, "metadata", id)
                        except KeyError:
                            pass

                    # title
                    if soup.title:
                        title = soup.title.get_text().encode("ascii", "replace").decode().lower()
                        self._process_data(title, "title", id)

                    # bold
                    bolded = soup.find("b")
                    if bolded:
                        bolded = bolded.get_text().encode("ascii", "replace").decode().lower()
                        self._process_data(bolded, "bolded", id)

                    # h1
                    h1 = soup.find("h1")
                    if h1:
                        h1 = h1.get_text().encode("ascii", "replace").decode().lower()
                        self._process_data(h1, "h1", id)

                    # h2
                    h2 = soup.find("h2")
                    if h2:
                        h2 = h2.get_text().encode("ascii", "replace").decode().lower()
                        self._process_data(h2, "h2", id)

                    # h3
                    h3 = soup.find("h3")
                    if h3:
                        h3 = h3.get_text().encode("ascii", "replace").decode().lower()
                        self._process_data(h3, "h3", id)

        print("finished parsing")
        self.calculate_tfidf()


    def calculate_tfidf(self):
        for id in self.tf:
            for term in self.tf[id]:
                tf  = math.log(self.tf[id][term], 10) + 1
                idf = math.log(self.num_docs/self.df[term], 10)
                self.idfs[term] = idf
                self.words[term][id]["tfidf"] = tf * idf


    def write_data(self):
        try:
            # ! https://pymongo.readthedocs.io/en/stable/examples/bulk.html
            self.db.example.bulk_write([InsertOne(
                {"_id": term, "idf": self.idfs[term], "docId": self.words[term]}) for term in self.words])
            print("finished writing")
            # db.example.bulk_write([InsertOne({"_id": term, "urls": unique_words[term]}) for term in unique_words])
        except BulkWriteError as bwe:
            print(bwe.details)


    def _process_data(self, text, tag, id):
        tokens = word_tokenize(text)
        for token in tokens:
            if (len(token) >= 3 and token not in STOPWORDS and token.isalnum()) or token.isdigit():
                term = self.lemmatizer.lemmatize(token)
                if token == "research":
                    print(term, token)
                    
                if term in self.tf[id]:
                    self.words[term][id][tag] = True


if __name__ == "__main__": 
    i = IndexConstructor()
    i.construct_index()
    i.write_data()