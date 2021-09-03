"""
cv_score.py
"""

# 3rd party imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CVScoreHolePercentage(object):

    def __init__(self, model, cv_index, train_index, test_index, x, y, hole_percentage):

        """
        This class computes the cross-validation score when a random portion of the test hole
        is placed back in the training dataset.

        :param model:
        :param cv_index:
        :param train_index:
        :param test_index:
        :param x:
        :param y:
        :param hole_percentage: percentage of test hole to randomly place back in the training dataset.
        """

        # Set parameters
        self.model = model
        self.cv_index = cv_index
        self.train_index = train_index
        self.test_index = test_index
        self.x = x
        self.y = y
        self.hole_percentage = hole_percentage

        # Remove hole_percentage from test hole and add to training data
        if self.hole_percentage > 0:
            test, train = train_test_split(self.test_index, test_size=hole_percentage, random_state=0)
            self.train_index.extend(train)
            self.test_index = test
        else:
            pass

        # Set attributes
        self.x_train = self.x.loc[self.train_index, :]
        self.y_train = self.y[self.train_index]
        self.x_test = self.x.loc[self.test_index, :]
        self.y_test = self.y[self.test_index]
        self.y_test_pred = None
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.mean_absolute_error = None
        self.r2_score = None

        # Train model
        self.model.fit(X=self.x_train, y=self.y_train)

        # Compute CV score
        self._compute_score()

    def get_cv_score(self):
        """Return a dictionary of final scores."""
        return {'cv_index': self.cv_index,
                'mean_squared_error': self.mean_squared_error,
                'root_mean_squared_error': self.root_mean_squared_error,
                'mean_absolute_error': self.mean_absolute_error,
                'r2_score': self.r2_score}

    def _compute_score(self):
        """Compute the CV score."""
        # Get test prediction
        self.y_test_pred = self.model.predict(self.x_test)

        # Compute metrics
        self.mean_squared_error = mean_squared_error(y_pred=self.y_test_pred, y_true=self.y_test)
        self.root_mean_squared_error = np.sqrt(self.mean_squared_error)
        self.mean_absolute_error = mean_absolute_error(y_pred=self.y_test_pred, y_true=self.y_test)
        self.r2_score = r2_score(y_pred=self.y_test_pred, y_true=self.y_test)
