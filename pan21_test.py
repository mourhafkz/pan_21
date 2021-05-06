import argparse
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
stop_words_en = stopwords.words('english')
stop_words_es = stopwords.words('spanish')
nlpES = spacy.load('es_core_news_sm')





"""
######-------  Data loading and xml writing Functions adopted from https://github.com/pan-webis-de/ashraf19 --######
"""


def iter_docs(author):
    author_attr = author.attrib
    doc_dict = author_attr.copy()
    #    print(doc_dict)
    doc_dict['text'] = [' '.join([doc.text for doc in author.iter('document')])]

    return doc_dict


def create_data_frame(input_folder):
    os.chdir(input_folder)
    all_xml_files = glob.glob("*.xml")
    truth_data = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'author'], engine='python')

    temp_list_of_DataFrames = []
    text_Data = pd.DataFrame()
    for file in all_xml_files:
        etree = ET.parse(file)  # create an ElementTree object
        doc_df = pd.DataFrame(iter_docs(etree.getroot()))
        doc_df['author_id'] = file[:-4]
        temp_list_of_DataFrames.append(doc_df)
    text_Data = pd.concat(temp_list_of_DataFrames, axis=0)

    data = text_Data.merge(truth_data, on='author_id')
    return data


def writefiles(data, output, lang):
    try:
        os.chdir(output)
    except Exception as e:
        print(e)
    try:
        os.mkdir(lang)
    except Exception as e:
        print(e)
    try:
        os.chdir(lang)
    except Exception as e:
        print(e)
    for index, row in data.iterrows():
        print(row['author_id'], row['lang'], row['author'])
        root = ET.Element("author", id=row['author_id'], lang=row['lang'], type=str(row['author']))
        tree = ET.ElementTree(root)
        tree.write(row['author_id'] + ".xml")


def getArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
    parser.add_argument("-o", "--output", help="Ouput Directory Path", required=True)
    args = parser.parse_args()

    print("input {} output {} ".format(
        args.input,
        args.output,
    ))

    return args.input, args.output

"""
######----------  Preprocessing Functions English ------------######
"""

# loading hate speech intensity scale
dictionary = {}
# with open("D:/data/bad-hate-dictionary.txt", "r") as f:
with open(os.path.join('bad-hate-dictionary.txt'), 'r') as f:
    for line in f:
        s = line.strip().split(" ")
        dictionary[s[0]] = s[1]
f.close()


# function to replace multiple entries in a line in one go via regex
def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


# text preprocessing function
def en_preprocess(data):
    corpus = []
    for tweets in data:
        tweets_lowered = tweets.lower()  # lowercase
        tweet = multiple_replace(dictionary, tweets_lowered)  # hate speech flags
        tweet = re.sub(r'\s+[a-z]\s+', ' ', tweet)  # remove single characters like i and a
        tweet = re.sub(r'^[a-z]\s+', ' ', tweet)  # remove single characters at the beginning like i and a
        tweet = re.sub(r'\s+', ' ', tweet)  # remove extra spaces
        tweets_tokenized = word_tokenize(tweet)  # tokenize
        tweets_no_stopwords = [w for w in tweets_tokenized if w not in stop_words_en]  # remove stopwords
        corpus.append(' '.join(tweets_no_stopwords))
    return corpus


# search and count functions
def face_concerned(text): return len([c for c in text if c in '😕😟🙁☹😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😞😓😩😫🥱🙀😿'])
def face_negative(text): return len([c for c in text if c in '😤😡😠🤬😈👿💀☠😾'])
def people(text): return len([c for c in text if c in '👶🧒👦👧🧑👱👨🧔👩🧓👴👵🙍🙍‍♂️🙎🙅🙆💁🙋🧏🙇🤦🤦‍♂️🤷🤷‍♂️'])
def bad_words(text): return len([c for c in text.split() if c == "INSULT_WORD"])
def hate1_counts(text): return len([c for c in text.split() if c == "HATE_LVL1"])
def hate2_counts(text): return len([c for c in text.split() if c == "HATE_LVL2"])
def hate3_counts(text): return len([c for c in text.split() if c == "HATE_LVL3"])
def hate4_counts(text): return len([c for c in text.split() if c == "HATE_LVL4"])
def hate5_counts(text): return len([c for c in text.split() if c == "HATE_LVL5"])
def hate6_counts(text): return len([c for c in text.split() if c == "HATE_LVL6"])
def hateMisc_counts(text): return len([c for c in text.split() if c == "HATE_MISC"])
def user_count(text): return len([c for c in text.split() if c == "user"])
def hashtag_counts(text): return len([c for c in text.split() if c == "hashtag"])
def url_counts(text): return len([c for c in text.split() if c == "http "])


# collective count function
def counters(data):
    data['face_concerned'] = data['preprocessed_text'].apply(face_concerned)
    data['face_negative'] = data['preprocessed_text'].apply(face_negative)
    data['people'] = data['preprocessed_text'].apply(people)
    data['bad_words'] = data['preprocessed_text'].apply(bad_words)
    data['hate_lvl1'] = data['preprocessed_text'].apply(hate1_counts)
    # data['hate_lvl2'] = data['preprocessed_text'].apply(hate2_counts) # no effect on accuracy
    data['hate_lvl3'] = data['preprocessed_text'].apply(hate3_counts)
    data['hate_lvl4'] = data['preprocessed_text'].apply(hate4_counts)
    # data['hate_lvl5'] = data['preprocessed_text'].apply(hate5_counts) # no effect on accuracy
    # data['hate_lvl6'] = data['preprocessed_text'].apply(hate6_counts) # no effect on accuracy
    data['hate_MISC'] = data['preprocessed_text'].apply(hateMisc_counts)
    # data['user_count'] = data['preprocessed_text'].apply(user_count) # no effect on accuracy
    # data['hashtag_counts'] = data['preprocessed_text'].apply(hashtag_counts) # no effect on accuracy
    data['url_count'] = data['preprocessed_text'].apply(lambda x: len(re.findall('http\S+', x)))  # from ashraf2019
    data['space_count'] = data['preprocessed_text'].apply(lambda x: len(re.findall(' ', x)))  # from ashraf2019


"""
######----------  Preprocessing Functions Spanish ------------######
"""


def es_preprocess(data):
    corpus = []
    for tweets in data:
        tweets_lowered = tweets.lower()
        # Further tweet sanitation
        tweet = re.sub(r'\s+[a-z]\s+', ' ', tweets_lowered)  # remove single characters like i and a
        tweet = re.sub(r'^[a-z]\s+', ' ', tweet)  # remove single characters at the beginning like i and a
        tweet = re.sub(r'\srt\s+', '', tweet)  # remove extra spaces
        tweet = re.sub(r'#user#', ' #user# ', tweet)  # remove extra spaces
        tweet = re.sub(r'#url#', '', tweet)  # remove extra spaces
        tweet = re.sub(r'\s+', ' ', tweet)  # remove extra spaces
        tokenizedTweet = nlpES(tweet)
        processedTweet = []
        for l in tokenizedTweet:
            processedTweet.append(f"{l.lemma_}")
        tweets_no_stopwords = [w for w in processedTweet if w not in stop_words_es]
        corpus.append(' '.join(tweets_no_stopwords))
    return corpus


"""
######---------- unpickle and test English/Spanish ------------######
"""


def test_en(input_folder, output_folder, lang):
    input_fold = os.path.join(input_folder,lang)
    data = create_data_frame(input_fold)
    # loading data
    X, y = data['text'], data['author']
    # preprocessing
    training_data = pd.DataFrame()
    training_data['preprocessed_text'] = en_preprocess(X)
    counters(training_data)
    features = training_data.drop(['preprocessed_text'], axis=1)

    if training_data.isnull().values.any():
        training_data.isnull().values.any()
        training_data.fillna(0, inplace=True)
    try:
        os.chdir(root)
    except Exception as e:
        print(e)
    print(os.getcwd())
    model = pickle.load(open(os.path.join('models', lang, 'model'), 'rb'))
    author = model.predict(training_data.drop(['preprocessed_text'], axis=1))
    data['author'] = author
    data.loc[data.author == 'spreader'] = 'spreader'
    writefiles(data,output_folder,lang)


def test_es(input_folder, output_folder, lang):
    input_fold = os.path.join(input_folder, lang)
    # loading data
    data = create_data_frame(input_fold)
    X, y = data['text'], data['author']
    # preprocessing
    preprocessed = es_preprocess(X)
    try:
        os.chdir(root)
    except Exception as e:
        print(e)
    print(os.getcwd())
    model = pickle.load(open(os.path.join('models', lang, 'model'), 'rb'))
    vectorizer = pickle.load(open(os.path.join('vectorizers', lang, 'vectorizer'), 'rb'))
    X_test = vectorizer.transform(preprocessed).toarray()
    author = model.predict(X_test)
    data['author'] = author
    data.loc[data.author == 'spreader' ] = 'spreader'
    writefiles(data,output_folder,lang)




def main():
    global root

    root = os.getcwd()

    input_folder, output_folder = getArg()
    # input_folder, output_folder = 'D:/pan_data/', 'D:/pan_data/test/output'
    test_en(input_folder, output_folder, 'en')
    test_es(input_folder, output_folder, 'es')






if __name__ == "__main__":
    main()

#  we need to output and write xml files
