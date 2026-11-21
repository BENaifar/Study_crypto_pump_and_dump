"""Classifier factories with the original hyper-parameters. [orig: train_classifier.py:L67-L72]"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

MODEL_NAMES = ['Logistic Regression', 'GBDT', 'Random Forest', 'Naive Bayes']


def make_lr(c) -> LogisticRegression:
    return LogisticRegression(random_state=c["random_state"], C=c["C"])


def make_gbdt(c) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(learning_rate=c["learning_rate"], min_samples_split=c["min_samples_split"],
                                      min_samples_leaf=c["min_samples_leaf"], max_depth=c["max_depth"],
                                      max_features=c["max_features"], subsample=c["subsample"],
                                      random_state=c["random_state"])


def make_rf(c) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=c["n_estimators"], criterion=c["criterion"],
                                  random_state=c["random_state"])


def make_mnb(c) -> MultinomialNB:
    return MultinomialNB(alpha=c["alpha"])


def make_all(classifier_cfg) -> list:
    """Models in MODEL_NAMES order."""
    return [make_lr(classifier_cfg.lr), make_gbdt(classifier_cfg.gbdt), make_rf(classifier_cfg.rf),
            make_mnb(classifier_cfg.mnb)]
