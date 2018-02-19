from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


file_a1 = open("1.txt", 'r')

a1 = file_a1.read()
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

print(tempStemmer)
#print(wordsFiltered)