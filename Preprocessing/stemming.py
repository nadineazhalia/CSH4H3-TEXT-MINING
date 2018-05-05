from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

wordLemma = WordNetLemmatizer()
stemmer = PorterStemmer()
tempLemma = []
tempStemmer = []
sentence = "Bitcoin plunged as the cancellation of a technology upgrade prompted some users to switch out of the cryptocurrency, spooking speculators who had profited from a more than 800 percent surge this year."
sentence = sentence.split()
for word in sentence:
    tempStemmer.append(stemmer.stem(word))
    tempLemma.append(wordLemma.lemmatize(word))
print(tempLemma)
print(tempStemmer)