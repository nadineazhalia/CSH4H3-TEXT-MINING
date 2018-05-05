from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


file_a1 = open("1.txt", 'r')
file_a2 = open("2.txt", 'r')
file_a3 = open("3.txt", 'r')
file_a4 = open("4.txt", 'r')
file_a5 = open("5.txt", 'r')

all_file = [file_a1, file_a2, file_a3, file_a4, file_a5]
tf_dict = {}
idf_dict = {}
idf_dict_val = {}

for f in all_file:
    a1 = f.read()
    f.close()
    a1 = a1.split()
    a1 = [x.lower() for x in a1]
    stop = set(stopwords.words('english'))
    #words = word_tokenize(a1)
    wordsFiltered = []

    #print ([i for i in a1 if i not in stop])
    #print(a1)

    for w in a1:
        if w not in stop:
            wordsFiltered.append(w)

    stemmer = PorterStemmer()
    tempStemmer = []
    for w in wordsFiltered:
        tempStemmer.append(stemmer.stem(w))

    nama_var = "berita" + str(all_file.index(f) + 1)
    tf_dict[nama_var] = {}

    for wrd in tempStemmer:
        if wrd in tf_dict[nama_var].keys():
            tf_dict[nama_var][wrd] += 1
        else:
            tf_dict[nama_var][wrd] = 1
    
    for key in tf_dict[nama_var].keys():
        if key in idf_dict.keys():
            idf_dict[key] += 1
        else:
            idf_dict[key] = 1

    for key, val in idf_dict.items():
        idf_dict_val[key] = val/5
print(tf_dict)
print(idf_dict_val)

# Masukin ke EXCEL
import pandas as pd
df_tf = pd.read_excel("hasil tf-idf.xlsx", "TF")
print(df_tf.head())
