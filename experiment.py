import codecs
import ConfigParser
import glob
import os
from itertools import chain

import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sb
sb.plt.ioff()

from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle

from vector_space_models import PIPELINES
from sample_documents import SampleTexts

config = ConfigParser.ConfigParser()
config.read("config.txt")

def read_files(filenames):
    for filename in filenames:
        with codecs.open(filename, encoding='utf-8') as f:
            yield f.read()

# set a random seed for reproduceability
random_state = config.getint('other', 'random-state')
# De volgende twee initialisaties van de klasse SampleTexts stellen
# twee iterators op waarmee random teksten van willekeurige lengte
# worden gegenereerd.
dutch = SampleTexts(config.get('documents', 'train_dutch'),
                    n_doc=config.getint('documents', 'n_documents'),
                    min_length=config.getint('documents', 'min_n_sentences'),
                    max_length=config.getint('documents', 'max_n_sentences'),
                    skip=config.getint('documents', 'skip'),
                    random_state=random_state)

flemish = SampleTexts(config.get('documents', 'train_flemish'),
                      n_doc=config.getint('documents', 'n_documents'),
                      min_length=config.getint('documents', 'min_n_sentences'),
                      max_length=config.getint('documents', 'max_n_sentences'),
                      skip=config.getint('documents', 'skip'),
                      random_state=random_state)

# initialiseer een vector-space model. Dit is een Pipeline waarmee
# de teksten worden ingeladen, getokeniseerd (naar woord of karakters)
# en vervolgens word geanalyseerd.
vectorizer = PIPELINES[config.get('vector-space', 'model')]

# de gekozen vectorizer maakt gebruik van default instellingen
# hier halen we de gekozen instellingen op uit het configuratiebestand
# en voegen die toe aan de vectorizer
parameters = {'tf__analyzer': config.get('vector-space', 'feature-type'),
              'tf__ngram_range': map(int, config.get('vector-space', 'ngram-range').split(','))}
if config.get('vector-space', 'model') == 'plm':
    parameters['plm__weight'] = config.getfloat('vector-space', 'lambda')
    parameters['plm__iterations'] = config.getint('vector-space', 'plm-iterations')
vectorizer.set_params(**parameters)

# als alles klaar is, kunnen we de vectorizer fitten (om alle gewichten ed
# te bepalen) en daarna kunnen we de gefitte vectorizer gebruiken om de data
# te "transformen". De methode fit_transform doet allebei in een keer.
X = vectorizer.fit_transform(chain(dutch, flemish))

# We hebben we array nodig van klasse-labels voor alle teksten. De klassen
# dutch en flemish hebben een teller (n) die bijhoudt hoeveel documenten er
# genenereerd zijn. Aangezien de documenten en de rijen in X, nog op volgorde
# staan kunnen we die tellingen rechtstreeks gebruiken om een array van
# corresponderende labels maken.
y = np.array([0] * dutch.n + [1] * flemish.n)

# We maken straks gebruik van een Stochastic Gradient Descent Classifier.
# Gezien de learning rate, is het verstandig om de teksten op voorhand al
# te shufflelen.
X, y = shuffle(X, y, random_state=random_state)

# als er geen test directory opgegeven is, gaan we ervanuit dat er getrained
# en getest moet worden op de training data.
if config.get('documents', 'test') == 'no':
    # met the functie train_test_split, verdelen we de data in een random
    # training en test deel.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
# anders maken lezen we alle test documenten in
else:
    X_train, y_train = X, y
    doc_ids = glob.glob(os.path.join(config.get('documents', 'test'), "*.txt"))
    X_test = vectorizer.transform(read_files(doc_ids))

# Als in het configuratiebestand feature-selectie op yes staat, passen
# we X^2 feature selectie toe op de data en nemen daarvan de x% beste
# features (aan te geven met de percentile parameter in de config)
if config.getboolean("vector-space", "feature-selection"):
    selector = SelectPercentile(
        chi2, percentile=config.getfloat("vector-space", "percentile"))
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

# nu is alles klaar en kunnen we de classifier initialiseren. Ik maak gebruik
# van een Stochastic Gradient Descent Classifier. De loss-functie bepaalt het
# type classifier: 'hinge' maakt een Linear SVC, 'log' een Logistic Regression
# classifier. 'n_iter' geeft het aantal iteraties per epoch aan. Het kan van
# belang zijn deze parameter hoger in te stellen als de resultaten niet
# consistent zijn.
classifier = SGDClassifier(n_iter=50, loss=config.get("classifier", "loss"),
                           shuffle=True, random_state=random_state)
# We fitten (trainen) de classifier als volgt:
classifier.fit(X_train, y_train)

if config.get('documents', 'test') == 'no':
    # nu is alles klaar om de classifier te testen op onze test set
    preds = classifier.predict(X_test)

    # de decision_function methode geeft de daadwerkelijke getallen terug
    # op basis waarvan de classificatie wordt gemaakt Dat kan handig zijn
    # later om een drempelwaarde te bepalen
    decisions = classifier.decision_function(X_test)
    print classification_report(y_test, preds)
    print "Area Under the Precision Recall Curve:",  average_precision_score(y_test, decisions)
    precision, recall, _ = precision_recall_curve(y_test, decisions)
    sb.plt.figure()
    sb.plt.plot(recall, precision)
    sb.plt.savefig("Precision-recall-curve.pdf")
else:
    decisions = classifier.decisions(X_test)
    preds = classifier.predict(X_test)
    for doc_id, decision, pred in sorted(zip(doc_ids, decisions, preds), key=lambda i: i[1]):
        print 'Document:', doc_id, "Score: %.4f, Prediction: %s" % (
            decision, 'NL' if pred == 0 else 'B')
