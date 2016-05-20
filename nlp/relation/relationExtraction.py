# import re
# import nltk
#
# IN = re.compile(r'.*\bin\b(?!\b.+ing)')
# for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
#     for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
#         print nltk.sem.rtuple(rel)


import nltk
import re
from nltk.sem import extract_rels,rtuple

#billgatesbio from http://www.reuters.com/finance/stocks/officerProfile?symbol=MSFT.O&officerId=28066
with open('test.txt', 'r') as f:
    sample = f.read().decode('utf-8')

sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

# here i changed reg ex and below i exchanged subj and obj classes' places
OF = re.compile(r'.*\bof\b.*')

for i, sent in enumerate(tagged_sentences):
    sent = nltk.ne_chunk(sent) # ne_chunk method expects one tagged sentence
    rels = extract_rels('PER', 'ORG', sent, corpus='ace', pattern=OF, window=7) # extract_rels method expects one chunked sentence
    for rel in rels:
        print('{0:<5}{1}'.format(i, rtuple(rel)))