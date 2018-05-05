import xml.etree.ElementTree as ET
import glob
import os
import numpy as np
import re
import string
from collections import OrderedDict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#
path = './'
a = glob.glob(os.path.join(path, '*.xml'))
tandaBaca = set(string.punctuation)
tandaBaca.add("--")
tandaBaca.add("...")
tandaBaca.add("-")
tandaBaca.add("''")
tandaBaca.add("``")
tandaBaca.add("'")
tandaBaca.add("**")
stopWords = set(stopwords.words('english'))
stopWords.add("'s")
stopWords.add("'re")
stopWords.add("'m")
stopWords.add("'ll")
stopWords.add("'ve")
stopWords.add("'d")

semuanya_train = []
all_doc_train = []
kataunik_train = []

for filename in glob.glob(os.path.join(path, 'Training set\*.xml')):
    idx_train = filename.translate('.xml')
    idx_train = idx_train.translate('\\')
    listKata_train = []
    tree_train = ET.parse(filename)
    root_train = tree_train.getroot()

    headline_train = root_train.findall('./headline')
    teks_train = []
    hl_train = []
    for elem_train in headline_train:
        hl_train.append(elem_train.text)

    for elem_train in root_train:
        for subelem_train in elem_train:
            teks_train.append(subelem_train.text)
    t_train = ""
    for i_train in teks_train:
        t_train += str(i_train)+" "
    t_train = t_train.lower()
    hl_train = str(hl_train).lower()

    #t_train = re.sub(r'[^a-zA-Z]', ' ', t_train)
    t_train = re.sub(r'\d+', '', t_train)

    token_train = word_tokenize(t_train)
    token_train = list(OrderedDict.fromkeys(token_train))
    token_train = sorted(token_train, key=str.lower)
    filter = []
    for w_train in token_train:
        if w_train not in stopWords:
            if w_train not in tandaBaca:
                filter.append(w_train)

    token_train = filter
    ps_train = PorterStemmer()
    for kt_train in token_train:
        # print(ps.stem(kt))
        listKata_train.append(ps_train.stem(kt_train))
    listKata_train = list(OrderedDict.fromkeys(listKata_train))
    all_doc_train.append(t_train)
    semuanya_train.append((id, hl_train, t_train, token_train, listKata_train))

    kataunik_train.append(listKata_train)

all_kata_train = []
for i in kataunik_train:
    for j in i:
        all_kata_train.append(str(j))
all_kata_train = sorted(all_kata_train, key=str.lower)
all_kata_unik_train = list(OrderedDict.fromkeys(all_kata_train))

label = np.loadtxt("Training set.txt", dtype='str', delimiter=' ')
labelTrain = []
for i in label:
    labelTrain.append(i[1])

label = np.loadtxt("Testing set.txt", dtype='str', delimiter=' ')
labelTest = []
for i in label:
    labelTest.append(i[1])

semuanya_test = []
all_doc_test = []
kataunik_test = []

for filename in glob.glob(os.path.join(path, 'Testing set\*.xml')):
    idx_test = filename.translate('.xml')
    idx_test = idx_test.translate('\\')
    listKata_test = []
    tree_test = ET.parse(filename)
    root_test = tree_test.getroot()

    headline_test = root_test.findall('./headline')
    teks_test = []
    hl_test = []
    for elem_test in headline_test:
        hl_test.append(elem_test.text)

    for elem_test in root_test:
        for subelem_test in elem_test:
            teks_test.append(subelem_test.text)
    t_test = ""
    for i_test in teks_test:
        t_test += str(i_test)+" "
    t_test = t_test.lower()
    hl_test = str(hl_test).lower()

    t_test = re.sub(r'\d+', '', t_test)
    token_test = word_tokenize(t_test)
    token_test = list(OrderedDict.fromkeys(token_test))
    token_test = sorted(token_test, key=str.lower)
    filter = []
    for w_test in token_test:
        if w_test not in stopWords:
            if w_test not in tandaBaca:
                filter.append(w_test)

    token_test = filter
    ps_test = PorterStemmer()
    for kt_test in token_test:
        # print(ps.stem(kt))
        listKata_test.append(ps_test.stem(kt_test))
    listKata_test = list(OrderedDict.fromkeys(listKata_test))
    all_doc_test.append(t_test)
    semuanya_test.append((id, hl_test, t_test, token_test, listKata_test))

    kataunik_test.append(listKata_test)

all_kata_unik_test = []
for i in kataunik_test:
    for j in i:
        all_kata_unik_test.append(str(j))
all_kata_unik_test = sorted(all_kata_unik_test, key=str.lower)
all_kata_unik_test = list(OrderedDict.fromkeys(all_kata_unik_test))

dataFIX = all_kata_unik_test
for i in all_kata_unik_train:
    if i not in dataFIX:
        dataFIX.append(i)

##########################################################################
corpuses = all_doc_train + all_doc_test

# Menghitung TF IDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpuses).toarray()
tfidf_train = tfidf[:577]
tfidf_test = tfidf[577:]

# Klasifikasi MNB
clf_mnb = MultinomialNB()
clf_mnb.fit(tfidf_train, labelTrain)
pred_mnb = clf_mnb.predict(tfidf_test)
acc_test_mnb = metrics.accuracy_score(labelTest, pred_mnb)
print("Akurasi MNB: ", acc_test_mnb*100)

# Klasifikasi KNN
clf_knn = KNeighborsClassifier(n_neighbors=13, n_jobs=-1)
clf_knn.fit(tfidf_train, labelTrain)
pred_knn = clf_knn.predict(tfidf_test)
acc_test_knn = metrics.accuracy_score(labelTest, pred_knn)
print("Akurasi KNN: ", acc_test_knn*100)

##########################################################################
