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
import csv
from difflib import SequenceMatcher


def string_similarity(s1, s2):
    # Create a SequenceMatcher object with the two strings
    sequence_matcher = SequenceMatcher(None, s1, s2)

    # Get the ratio of similarity between the two strings
    similarity_ratio = sequence_matcher.ratio()

    # Calculate the percentage of matching
    matching_percentage = similarity_ratio * 100

    return matching_percentage


# ---------------------------------------------------------------------------

def Database_checker(url):
    data = []

    # Specify the path to your CSV file
    csv_file = "History.csv"

    # Open and read the CSV file
    with open(csv_file, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # Iterate through the rows and create tuples with columns 1
        for row in csv_reader:
            if len(row) <= 4:
                col1 = row[0]  # First column
                col4 = row[1]
                data.append(col1)
                data.append(col4)

    # print(data)

    if url in data:
        index = data.index(url)
        print(data[index + 1])
        return 0
    else:
        return 1


def database_updater(url, clkbat_y_n):
    # Define the data you want to append
    data_to_append = [url, clkbat_y_n]

    # Define the CSV file path
    csv_file_path = "History.csv"

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)

        # Write the data to the CSV file
        csv_writer.writerow(data_to_append)


# ---------------------------------------------------------------------------------------------------------------
def web_name_check(url):
    # Replace 'your_file.csv' with the path to your CSV file
    csv_file = 'Web_Scrapped_websites.csv'

    # Replace 'column_name' with the name of the column you want to extract unique values from
    column_name = 'Website'

    # Initialize an empty set to store unique values
    unique_values = set()
    updated_values = set()

    # Open the CSV file for reading
    with open(csv_file, 'r', newline='', encoding='latin1') as file:
        reader = csv.DictReader(file)

        # Iterate through each row in the CSV
        for row in reader:
            # Add the value in the specified column to the set
            unique_values.add(row[column_name])

    # Convert the set to a list if needed
    unique_values_list = list(unique_values)

    # Now, unique_values_list contains the unique values from the specified column
    # print(unique_values_list)

    for value in unique_values_list:
        parts = value.split('.')

        extracted_string = parts[1]
        updated_values.add(extracted_string)
        updated_list = list(updated_values)

    updated_list.append('livestrong')

    for i in range(len(updated_list)):
        if (string_similarity(updated_list[i], url) >= 85.00 and string_similarity(updated_list[i], url) != 100.00):
            return 0
    return 1


# ------------------------------------------------------------------------------------------------------------
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

        # print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

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
        # train_classifier(docs1)
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
url1 = 'https://www.daijiworld.com/news/newsDisplay?newsID=1009705'

# Not misleading
# url1 = 'https://www.indiatoday.in/business/story/ferrari-to-accept-crypto-currencies-as-payment-for-its-cars-in-us-europe-next-2449083-2023-10-14'

if web_name_check((url1.split('.'))[1]):
    if Database_checker(url1):
        response = requests.get(url1)
        html_content = response.content

        # Remove the ads.
        html_content = remove_ads(html_content)

        soup = BeautifulSoup(html_content, 'lxml')
        header = soup.find('h1').text.replace('\n', '')
        header_ans = extracting(header)
        # print(header_ans)

        paragraphs = soup.find_all('p')

        paragraph = ''

        for temp in paragraphs:
            paragraph = paragraph + (temp.text.replace('\n', ''))
        # print(paragraph)
        body_ans = extracting(paragraph)
        # print(body_ans)

        if header_ans == body_ans:
            print("Its not a misleading clickbait")
            database_updater(url1, 'Its not a misleading clickbait')
        else:
            print("Its a misleading clickbait")
            database_updater(url1, 'Its a misleading clickbait')

else:
    print("Its a misleading clickbait")
    database_updater(url1, 'Its a misleading clickbait')
