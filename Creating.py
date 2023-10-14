import os
import random
import string

import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
from bs4 import BeautifulSoup

import requests


def extracting(url_data):
    stop_words = set(stopwords.words('english'))
    stop_words.add('said')
    stop_words.add('mr')

    BASE_DIR = 'C:/Users/Nagathejas/Documents/Data set'
    LABELS = ['WELLNESS', 'POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 'PARENTING', 'FOOD & DRINK',
              'WORLD NEWS', 'BUSINESS', 'SPORTS']

    def setup_docs():
        # Initialize an empty list to store the tuples
        data = []

        # Specify the path to your CSV file
        csv_file = "temp.csv"

        # Open and read the CSV file
        with open(csv_file, "r", encoding='utf-8') as file:
            csv_reader = csv.reader(file)

            # Skip the header row if it exists
            next(csv_reader, None)

            # Iterate through the rows and create tuples with columns 1 and 4
            for row in csv_reader:
                if len(row) >= 4:
                    col1 = row[0]  # First column
                    col4 = row[3]  # Fourth column
                    data.append((col1, col4))
        return data

    def clean_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text

    def get_tokens(text):
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if not t in stop_words]
        return tokens

    def print_frequency_dist(docs):
        tokens = defaultdict(list)

        for doc in docs:
            doc_label = doc[0]

            doc_text = clean_text(doc[1])

            doc_tokens = get_tokens(doc_text)

            tokens[doc_label].extend(doc_tokens)

        for category_label, category_tokens in tokens.items():
            print(category_label)
            fd = FreqDist(category_tokens)
            print(fd.most_common(50))

    # model training
    def get_splits(docs):

        # scramble docs
        random.shuffle(docs)
        X_train = []  # training documents
        y_train = []  # corresponding training labels
        y_test = []
        X_test = []  # test #correspoding documents test label
        pivot = int(.80 * len(docs))
        for i in range(0, pivot):
            X_train.append(docs[i][1])
            y_train.append(docs[i][0])

        for i in range(pivot, len(docs)):
            X_test.append(docs[i][1])
            y_test.append(docs[i][0])
        return X_train, X_test, y_train, y_test

    def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_tfidf)

        precision = metrics.precision_score(y_test, y_pred, average='micro')
        recall = metrics.recall_score(y_test, y_pred, average='micro')
        f1 = metrics.f1_score(y_test, y_pred, average='micro')

        print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

    def train_classifier(docs):
        X_train, X_test, y_train, y_test = get_splits(docs)

        # the objects that turns text into vectors
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')
        # create doc_term matrix
        dtm = vectorizer.fit_transform(X_train)

        # train naive bayes classifier

        naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

        evaluate_classifier("Naive Bayes\TRAIN\t", naive_bayes_classifier, vectorizer, X_train, y_train)
        evaluate_classifier("Naive Bayes\TEST\t", naive_bayes_classifier, vectorizer, X_test, y_test)

        # store the classifier
        clf_filename = 'naive_bayes_classifier.plk'
        pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

        # also store the vectorizer so we can transform new data
        vec_filename = 'count_vectorizer.pkl'
        pickle.dump(vectorizer, open(vec_filename, 'wb'))

    def classify(text):
        # load classifier
        clf_filename = 'naive_bayes_classifier.plk'
        nb_clf = pickle.load(open(clf_filename, 'rb'))

        # vectorize the new text
        vec_filename = 'count_vectorizer.pkl'
        vectorizer = pickle.load(open(vec_filename, 'rb'))

        pred = nb_clf.predict(vectorizer.transform([text]))

        return pred[0]

    if __name__ == '__main__':
        docs1 = []
        docs1 = setup_docs()

        # print_frequency_dist(docs1)
        train_classifier(docs1)
        # deployment in production

        return classify(url_data)


def remove_ads(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Identify the ads.
    ads = soup.find_all('script', {'class': ['ad', 'advertisement']}) + soup.find_all('iframe', {
        'class': ['ad', 'advertisement']}) + soup.find_all('div', {'class': ['ad', 'advertisement']})

    # Remove the ads.
    for ad in ads:
        ad.extract()

    return soup.prettify()


url1 = ''

# Misleading
#url1 = 'https://www.daijiworld.com/news/newsDisplay?newsID=1009705'

# Not misleading
url1 =  'https://www.indiatoday.in/movies/celebrities/story/aaditya-thackeray-breaks-silence-i-am-not-linked-to-sushant-singh-rajput-death-this-is-dirty-politics-1707762-2020-08-04'

response = requests.get(url1)
html_content = response.content

# Remove the ads.
html_content = remove_ads(html_content)

soup = BeautifulSoup(html_content, 'lxml')
header = soup.find('h1').text.replace('\n', '')
header_ans = extracting("header")
print(header_ans)

paragraphs = soup.find_all('p')

paragraph = ''

for temp in paragraphs:
    paragraph = paragraph + (temp.text.replace('\n', ''))
body_ans = extracting("paragraph")
print(body_ans)

if header_ans == body_ans:
    print("Its not a misleading clickbait")
else:
    print("Its a misleading clickbait")
