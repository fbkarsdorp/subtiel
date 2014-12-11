import codecs

from itertools import islice

import numpy as np


class SampleTexts(object):
    """Text generator. Takes as input a single text file. Provides an iterator
    over samples from this text file with random lengths.
    Usage:

    >>> text_generator = SampleTexts(filename, 100, 10000)

    >>> for text in text_generator:
    >>>    print text
    """
    def __init__(self, filename, n_doc=None, min_length=10, max_length=100,
                 random_state=None, skip=1000):
        self.filename = filename
        self.min_length = min_length
        self.max_length = max_length
        self.rnd = np.random.RandomState(random_state)
        self.n_doc = n_doc
        self.skip = skip
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
                if self.skip:
                    # move the cursor n (specified by self.skip) sentences further
                    islice(infile, self.skip)
