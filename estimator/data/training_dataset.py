"""
training_dataset.py
"""

# 3rd party imports
import os
import pickle
import numpy as np


class TrainingDataset:

    """Training dataset for machine learning."""

    def __init__(self, train_geochem_index, x_train, y_train, hole_id_train, depth_train, train_holes,
                 test_geochem_index, x_test, y_test, hole_id_test, depth_test, test_holes):

        # Set parameters
        self.train_geochem_index = train_geochem_index
        self.x_train = x_train
        self.y_train = y_train
        self.hole_id_train = hole_id_train
        self.depth_train = depth_train
        self.train_holes = train_holes
        self.test_geochem_index = test_geochem_index
        self.x_test = x_test
        self.y_test = y_test
        self.hole_id_test = hole_id_test
        self.depth_test = depth_test
        self.test_holes = test_holes

        # Check train shapes
        assert len(train_geochem_index) == x_train.shape[0]
        assert x_train.shape[0] == y_train.shape[0]
        assert hole_id_train.shape[0] == y_train.shape[0]
        assert all([train_hole in np.unique(hole_id_train.values) for train_hole in train_holes])
        assert np.all(train_geochem_index.index.values == x_train.index.values)
        assert np.all(train_geochem_index.index.values == y_train.index.values)
        assert np.all(train_geochem_index.index.values == hole_id_train.index.values)
        assert np.all(x_train.index.values == y_train.index.values)

        # Check test shapes
        assert len(test_geochem_index) == x_test.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert hole_id_test.shape[0] == y_test.shape[0]
        assert all([train_hole in np.unique(hole_id_train.values) for train_hole in train_holes])
        assert np.all(test_geochem_index.index.values == x_test.index.values)
        assert np.all(test_geochem_index.index.values == y_test.index.values)
        assert np.all(test_geochem_index.index.values == hole_id_test.index.values)
        assert np.all(x_test.index.values == y_test.index.values)

        # Check train-test borehole separation
        assert any([train_hole not in test_holes for train_hole in train_holes])

        # Generate CV folds
        self.cv_folds = self._generate_cv_folds()

    def save(self, path):
        """Pickle data model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _generate_cv_folds(self):
        """Generate leave-one-hole-out CV folds."""
        # Dictionary for CV folds
        cv_folds = dict()

        # Loop through train holes
        for index, test_hole in enumerate(self.train_holes):

            # Train holes
            train_holes = [hole_id for hole_id in self.train_holes if hole_id != test_hole]

            # Add fold indices
            cv_folds[index] = {
                'test_hole': test_hole,
                'train_holes': train_holes,
                'test_index': self.hole_id_train[self.hole_id_train == test_hole].index.tolist(),
                'train_index': self.hole_id_train[~self.hole_id_train.isin([test_hole])].index.tolist()
            }

        return cv_folds
