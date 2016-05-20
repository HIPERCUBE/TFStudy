# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# URL : http://khanrc.tistory.com/entry/Text-Mining-Tutoral-20-newsgroups
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Loading the 20 newsgroups dataset
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
from sklearn.datasets import fetch_20newsgroups

print '# Loading the 20 newsgroups dataset'
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Extracting features from text files
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
print '# Extracting features from text files'
from sklearn.feature_extraction.text import CountVectorizer

# tokenizing text with scikit-learn
print '# tokenizing text with scikit-learn'
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print X_train_counts

print count_vect.vocabulary_.get('algorithm')
print count_vect.vocabulary_['algorithm']
print count_vect.__class__
print count_vect.vocabulary_.__class__

# From occurrences to frequencies
print  '# From occurrences to frequencies'
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print X_train_counts.shape
print X_train_tf.shape
print X_train_tfidf.shape

print X_train_counts[0]
print X_train_tf[0]
print X_train_tfidf[0]

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Training a classifier
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
print '# Training a classifier'
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print "{} => {}".format(doc, twenty_train.target_names[category])

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Building a pipeline
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
print '# Building a pipeline'
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Evaluation of the performance on the test set
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
print '# Evaluation of the performance on the test set'
import numpy as np

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)

print len(docs_test)
print "Naive Bayes Accuracy : ", np.mean(predicted == twenty_test.target)

# SVM
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
_ = text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
print "Support Vector Machine : ", np.mean(predicted == twenty_test.target)

# Analyze result
print '# Analyze result'
from sklearn import metrics

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print metrics.confusion_matrix(twenty_test.target, predicted)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Parameter tuning using grid search
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
print '# Parameter tuning using grid search'
from sklearn.grid_search import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

print twenty_train.target_names[gs_clf.predict(['God is love'])]

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print "%s: %r" % (param_name, best_parameters[param_name])

print 'score: ', score