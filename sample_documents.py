import codecs
import sys

from itertools import islice, chain

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


class SampleTexts(object):
    """Text generator. Takes as input a single text file. Provides an iterator
    over samples from this text file with random lengths.
    Usage:

    >>> text_generator = SampleTexts(filename, 100, 10000)

    >>> for text in text_generator:
    >>>    print text
    """
    def __init__(self, filename, n_doc=None, min_length=10, max_length=100,
                 random_state=None):
        self.filename = filename
        self.min_length = min_length
        self.max_length = max_length
        self.rnd = np.random.RandomState(random_state)
        self.n_doc = n_doc
        self.n = 0

    def __iter__(self):
        with codecs.open(self.filename, encoding='utf-8') as infile:
            while True:
                text = ' '.join(
                islice(infile, self.rnd.randint(self.min_length, self.max_length)))
                if not text or (self.n_doc != None and self.n >= self.n_doc):
                    raise StopIteration
                yield text
                self.n += 1

# set a random seed for reproduceability
random_state = 2014
# setup a sampler for dutch and flemish
dutch = SampleTexts(sys.argv[1], n_doc=10, random_state=random_state)
flemish = SampleTexts(sys.argv[2], n_doc=10, random_state=random_state)
# initialize the vector space model, here I use a simple tfidf vectorizer
# but we could also use the sparse PLM implementation
vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', min_df=1)
# fit and transform the data according to the vector space model
X = vectorizer.fit_transform(chain(dutch, flemish))
# we need an array of labels corresponding to each document:
y = np.array([0] * dutch.n + [1] * flemish.n)
# it is probably a good idea to shuffle the data first (depends on our classifier)
X, y = shuffle(X, y, random_state=random_state)
# now set up a classifier (I'm using the LinearSVC here), and train (fit) it:
classifier = LinearSVC(random_state=random_state)
classifier.fit(X, y)
# now we can use the classifier to predict stuff! yay!
