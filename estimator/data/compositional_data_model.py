"""
compositional_data_model.py
"""

# 3rd party imports
import pickle
import numpy as np
import pandas as pd


class CompositionalDataModel(object):

    """
    Data model for compositional data.

    Parameters
    ----------
    meta_data: pandas dataframe
        DataFrame of supplementary data (hole_id, lab method, lithology, etc.).
    comp_data_ppm : pandas dataframe
        DataFrame of compositional data in units of ppm.
    """

    def __init__(self, meta_data, comp_data_ppm):

        # Check DataFrame shapes
        assert meta_data.shape[0] == comp_data_ppm.shape[0]
        assert np.all(meta_data.index == comp_data_ppm.index)

        # Set input values
        self.meta_data = meta_data
        self.comp_data_ppm = comp_data_ppm

        # Set future values
        self.comp_data_closed = None
        self.comp_data_clr = None
        self.geo_other_data = None

        # Close compositional data
        self.compute_closure()

        # Compute centered log ratio (clr) transformation
        self.compute_clr()

    def compute_closure(self):

        # Close compositions
        self.comp_data_closed = self.comp_data_ppm.apply(lambda row: row / np.sum(row), axis=1)

        # Replace infinite values with NaN
        self.comp_data_closed = self.comp_data_closed.replace([np.inf, -np.inf], np.nan)

        # Drop any row with one NaN value
        self.comp_data_closed = self.comp_data_closed.dropna(how='any')

        # Get meta data
        self.meta_data = self.meta_data.loc[self.comp_data_closed.index, :]

        # Update ppm
        self.comp_data_ppm = self.comp_data_ppm.loc[self.comp_data_closed.index, :]

        assert np.all(self.meta_data.index == self.comp_data_closed.index)
        assert np.all(self.comp_data_ppm.index == self.comp_data_closed.index)

    def compute_clr(self):

        # Transform compositions
        self.comp_data_clr = pd.DataFrame(index=self.meta_data.index,
                                          data=self.clr(composition=self.comp_data_closed.values),
                                          columns=self.comp_data_closed.columns)

        # Replace infinite values with NaN
        self.comp_data_clr = self.comp_data_clr.replace([np.inf, -np.inf], np.nan)

        # Drop any row with one NaN value
        self.comp_data_clr = self.comp_data_clr.dropna(how='any')

        # Get meta data
        self.meta_data = self.meta_data.loc[self.comp_data_clr.index, :]

        # Update ppm
        self.comp_data_ppm = self.comp_data_ppm.loc[self.comp_data_clr.index, :]

        # Update closed
        self.comp_data_closed = self.comp_data_closed.loc[self.comp_data_clr.index, :]

        assert np.all(self.meta_data.index == self.comp_data_clr.index)
        assert np.all(self.comp_data_ppm.index == self.comp_data_clr.index)
        assert np.all(self.comp_data_closed.index == self.comp_data_clr.index)

    def get_data_model(self, index):
        """Return geochemistry matching density indices."""
        geo_other_data = self.geo_other_data.dropna()
        meta_data = self.meta_data.iloc[index, :]
        comp_data_ppm = self.comp_data_ppm.iloc[index, :]
        comp_data_closed = self.comp_data_closed.iloc[index, :]
        comp_data_clr = self.comp_data_clr.iloc[index, :]

        return meta_data, comp_data_ppm, comp_data_closed, comp_data_clr, geo_other_data

    def save_data_model(self, path):
        """Pickle data model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def clr(composition):
        """Performs centre log ratio transformation."""
        closed = composition / np.sum(composition)
        l_mat = np.log(closed)
        gm = l_mat.mean(axis=-1, keepdims=True)
        return (l_mat - gm).squeeze()
