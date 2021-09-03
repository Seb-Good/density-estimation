"""
qaqc_plotting_tools.py
"""

# 3rd party imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact

# Local imports
from estimator import WORKING_PATH
from estimator.data.compositional_data_model import CompositionalDataModel


def plot_lod(df, figure_size):

    """
    Bar plot of data count above and below the detection limit for each element.

    Parameters
    ----------
    df : pandas dataframe
        Original DataFrame of compositional data. Values below detection must be set to NaN. Columns must only
        include compositional data (No hole ID, from, to, etc..).
    figure_size : tuple
        (width_size, height_size)
    """

    # Calculate the percentage of data below detection for each element
    below_detection_count = [str(int(np.round(val / df.shape[0] * 100))) + ' %' for val in df.isnull().sum().values]

    # Calculate barplot parameters
    n = df.shape[1]
    ind = np.arange(n)  # The x locations for the groups
    width = 0.75        # The width of the bars: can also be len(x) sequence

    # Setup plot
    fig, ax1 = plt.subplots(figsize=figure_size)

    # plot >LOD
    p1 = plt.bar(ind, df.notnull().sum().values, width, color='#0870FF')

    # plot <=LOD
    p2 = plt.bar(ind, df.isnull().sum().values, width, bottom=df.notnull().sum().values, color='#FF1B0F')

    plt.legend((p1[0], p2[0]), ('>LOD', '<=LOD'), loc='center top', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)

    # Settings Y-Axis
    plt.yticks(fontsize=14)
    plt.ylabel('Data Points', fontsize=20)
    plt.ylim([0, df.shape[0]])

    # Settings X-Axis 1
    ax1.set_xlabel('Elements', fontsize=20)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(df.columns)
    labels = ax1.get_xticklabels()
    plt.setp(labels, rotation=90, fontsize=14)
    ax1.set_xlim([-1, df.shape[1]])

    # Settings X-Axis 2
    ax2 = ax1.twiny()
    ax2.xaxis.set_tick_params(labeltop='on')
    ax2.set_xticks(ind)
    ax2.set_xticklabels(below_detection_count, fontsize=14)
    labels = ax2.get_xticklabels()
    plt.setp(labels, rotation=90, fontsize=14)
    ax2.set_xlim([-1, df.shape[1]])

    plt.show()


def plot_transformation_inspection(df, eps_save=False):

    """
    Histograms of data (unclosed, close, log10 transformed, centered log ratio transformed.

    Parameters
    ----------
    df : pandas dataframe
        DataFrame of compositional data. Log ratio expectation maximization imputation has been conducted. No missing
        values in data. Units are in ppm.
    eps_save :
    """

    # Get list of all unique elements
    unique_elements = df.columns

    # Perform transformations
    df1 = df.copy().dropna(how='any')
    df2 = df.apply(lambda row: row / np.sum(row), axis=1)
    df3 = np.log10(df.copy())
    df4 = pd.DataFrame(CompositionalDataModel.clr(composition=df2.values), columns=df.columns)

    def plot_widget(element_id):

        # Setup Plot
        fig = plt.figure(figsize=(16, 12))

        fig.subplots_adjust(wspace=0.15, hspace=0.4)

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        ax4 = plt.subplot2grid((2, 2), (1, 1))

        # Get Element
        element = unique_elements[element_id]

        # Histogram
        ax1.set_title('Element: ' + element + '\n' + 'unclosed', fontsize=20)

        # Plot Data
        ax1.hist(df1[element].dropna(how='any'), 100, facecolor='#444445', alpha=0.75, edgecolor=[0.7, 0.7, 0.7])

        # Plotting parameters
        ax1.set_xlabel(element + ', ppm', fontsize=20)
        ax1.set_ylabel('Count', fontsize=20)

        # Histogram
        ax2.set_title('Element: ' + element + '\n' + 'closed', fontsize=20)

        # Plot Data
        ax2.hist(df2[element].dropna(how='any'), 100, facecolor='#F7B318', alpha=0.75, edgecolor=[0.7, 0.7, 0.7])

        # Plotting parameters
        ax2.set_xlabel(element, fontsize=20)
        ax2.set_ylabel('Count', fontsize=20)

        # Histogram
        ax3.set_title('Element: ' + element + '\n' + 'log10( unclosed )', fontsize=20)

        # Plot Data
        ax3.hist(df3.loc[np.isfinite(df3[element]), element], 100,
                 facecolor='#444445', alpha=0.75, edgecolor=[0.7, 0.7, 0.7])

        # Plotting parameters
        ax3.set_xlabel(element, fontsize=20)
        ax3.set_ylabel('Count', fontsize=20)

        # Histogram
        ax4.set_title('Element: ' + element + '\n' + 'clr( closed )', fontsize=20)

        # Plot Data
        ax4.hist(df4.loc[np.isfinite(df4[element]), element], 100,
                 facecolor='#F7B318', alpha=0.75, edgecolor=[0.7, 0.7, 0.7])

        # Plotting parameters
        ax4.set_xlabel(element, fontsize=20)
        ax4.set_ylabel('Count', fontsize=20)

        if eps_save:
            plt.savefig(os.path.join(WORKING_PATH, 'figures', 'geochem_histogram.eps'), format='eps')

        plt.show()

    # Launch widget
    interact(plot_widget,
             element_id=(0, len(unique_elements)-1, 1))
