from __future__ import unicode_literals
import sys
import codecs
import nltk

def parse(file_path):
    data = None
    with codecs.open(file_path, 'r') as file:
        data = nltk.word_tokenize(file.read())
    return data

print parse(sys.argv[1])
