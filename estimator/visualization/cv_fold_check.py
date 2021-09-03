"""
cv_fold_check.py
"""

# 3rd party imports
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed


def plot_depth_profiles(training_dataset, density, cv_index, figure_size=(20, 10), comparison_plot_count=5):

    """Plot Setup"""
    fig, axs = plt.subplots(
        1, len(training_dataset.train_holes),
        figsize=figure_size,
    )
    fig.subplots_adjust(wspace=0.2)
    axs = axs.ravel()
    axt = plt.subplot2grid((1, len(training_dataset.train_holes)), (0, len(training_dataset.train_holes)-1))

    """Plot Comparison Columns"""
    # Loop through comparison axes
    for ax in range(len(training_dataset.train_holes) - 1):

        hole_id = training_dataset.cv_folds[cv_index]['train_holes'][ax]
        index = training_dataset.hole_id_train[training_dataset.hole_id_train == hole_id].index

        sdst_start = training_dataset.depth_train.loc[
            training_dataset.x_train.loc[index, 'SDST'][training_dataset.x_train.loc[index, 'SDST'] == 1].index[
                0], 'depth_from']
        sdst_end = training_dataset.depth_train.loc[
            training_dataset.x_train.loc[index, 'SDST'][training_dataset.x_train.loc[index, 'SDST'] == 1].index[
                -1], 'depth_to']

        rect_sdst = matplotlib.patches.Rectangle((2.9, sdst_start), 0.1, (sdst_end - sdst_start),
                                                 facecolor='m', edgecolor='k', lw=1, label='SDST')

        axs[ax].add_patch(rect_sdst)

        if training_dataset.x_train.loc[index, 'BSMT'][training_dataset.x_train.loc[index, 'BSMT'] == 1].shape[0] > 0:
            bsmt_end = training_dataset.depth_train.loc[
                training_dataset.x_train.loc[index, 'BSMT'][training_dataset.x_train.loc[index, 'BSMT'] == 1].index[
                    -1], 'depth_to']
            rect_bsmt = matplotlib.patches.Rectangle((2.9, sdst_end), 0.1, (bsmt_end - sdst_end),
                                                     facecolor='c', edgecolor='k', lw=1, label='BSMT')
            axs[ax].add_patch(rect_bsmt)

        axs[ax].set_title('Train\n{}\n{}'.format(hole_id.split(' ')[0], hole_id.split(' ')[1]), fontsize=12, y=1.03)
        axs[ax].plot(density['density'][density['plot_name'] == hole_id],
                     density['depth'][density['plot_name'] == hole_id],
                     '-', color=[0.7, 0.7, 0.7], label='Raw', lw=1)
        axs[ax].plot(training_dataset.y_train[index].values,
                     training_dataset.depth_train.loc[index, 'depth_mean'].values,
                     'ob', lw=2, label='Target')

        axs[ax].get_yaxis().set_visible(False)
        axs[ax].locator_params(nbins=4, axis='x')
        axs[ax].tick_params(labelsize=12)
        plt.setp(axs[ax].get_xticklabels(), rotation=90)
        axs[ax].xaxis.labelpad = 10
        axs[ax].set_xlim([2.3, 3])
        axs[ax].set_ylim([training_dataset.depth_train.loc[index, 'depth_mean'].min(),
                          training_dataset.depth_train.loc[index, 'depth_mean'].max()])
        axs[ax].set_xlabel('Density\ng/cm$^{3}$', fontsize=12)
        axs[ax].invert_yaxis()

    """Plot Target Column"""
    hole_id = training_dataset.cv_folds[cv_index]['test_hole']
    index = training_dataset.hole_id_train[training_dataset.hole_id_train == hole_id].index

    axt.set_title('Test\n{}\n{}'.format(hole_id.split(' ')[0], hole_id.split(' ')[1]), fontsize=12, y=1.03)

    sdst_start = training_dataset.depth_train.loc[
        training_dataset.x_train.loc[index, 'SDST'][training_dataset.x_train.loc[index, 'SDST'] == 1].index[
            0], 'depth_from']
    sdst_end = training_dataset.depth_train.loc[
        training_dataset.x_train.loc[index, 'SDST'][training_dataset.x_train.loc[index, 'SDST'] == 1].index[
            -1], 'depth_to']

    rect_sdst = matplotlib.patches.Rectangle((2.9, sdst_start), 0.1, (sdst_end - sdst_start),
                                             facecolor='m', edgecolor='k', lw=1, label='SDST')

    axt.add_patch(rect_sdst)

    if training_dataset.x_train.loc[index, 'BSMT'][training_dataset.x_train.loc[index, 'BSMT'] == 1].shape[0] > 0:
        bsmt_end = training_dataset.depth_train.loc[
            training_dataset.x_train.loc[index, 'BSMT'][training_dataset.x_train.loc[index, 'BSMT'] == 1].index[
                -1], 'depth_to']
        rect_bsmt = matplotlib.patches.Rectangle((2.9, sdst_end), 0.1, (bsmt_end - sdst_end),
                                                 facecolor='c', edgecolor='k', lw=1, label='BSMT')
        axt.add_patch(rect_bsmt)

    axt.plot(density['density'][density['plot_name'] == hole_id],
             density['depth'][density['plot_name'] == hole_id],
             '-', color=[0.7, 0.7, 0.7], label='Raw', lw=1)
    axt.plot(training_dataset.y_train[index].values,
             training_dataset.depth_train.loc[index, 'depth_mean'].values,
             'or', lw=2, label='Target')

    axt.get_yaxis().set_visible(False)
    axt.locator_params(nbins=4, axis='x')
    axt.tick_params(labelsize=12)
    plt.setp(axt.get_xticklabels(), rotation=90)
    axt.xaxis.labelpad = 10
    axt.set_xlabel('Density\ng/cm$^{3}$', fontsize=12)
    axt.set_xlim([2.3, 3])
    axt.set_ylim([training_dataset.depth_train.loc[index, 'depth_mean'].min(),
                  training_dataset.depth_train.loc[index, 'depth_mean'].max()])
    axt.invert_yaxis()

    plt.plot()


def cv_fold_check(training_dataset, density, figure_size=(20, 10), comparison_plot_count=5):

    # Launch widget
    interact(plot_depth_profiles,
             training_dataset=fixed(training_dataset),
             density=fixed(density),
             cv_index=(0, len(training_dataset.train_holes) - 1, 1),
             figure_size=fixed(figure_size),
             comparison_plot_count=fixed(comparison_plot_count))
