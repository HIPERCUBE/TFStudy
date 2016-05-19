from nltk.corpus import gutenberg  # Docs from project gutenberg.org
from nltk import regexp_tokenize
import nltk

files_en = gutenberg.fileids()  # Get file ids
doc_en = gutenberg.open('austen-emma.txt').read()

pattern = r'''(?x) ([A-Z]\.)+ | \w+(-\w+)* | \$?\d+(\.\d+)?%? | \.\.\. | [][.,;"'?():-_`]'''
tokens_en = regexp_tokenize(doc_en, pattern)

en = nltk.Text(tokens_en)

print(len(en.tokens))  # returns number of tokens (document length)
print(len(set(en.tokens)))  # returns number of unique tokens
en.vocab()
