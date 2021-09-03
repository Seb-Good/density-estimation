"""
depth_profiles_estimations.py
"""

# 3rd party imports
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed


def plot_depth_profiles(comp_data_old, meta_data_old, comp_data_new, meta_data_new,
                        density, density_merged, model, hole_index, figure_size=(20, 10), comparison_plot_count=5):

    # Format Input
    plot_name = model.dataset.test_holes[hole_index]
    hole_id = plot_name.split(' ')[0]
    density = density[density['plot_name'] == plot_name]
    density_merged = density_merged[density_merged['plot_name'] == plot_name]
    lab_method = density.loc[density.index[0], 'lab_method']

    if lab_method == 'old':
        meta_temp = meta_data_old[meta_data_old['hole_id'] == hole_id]
        comp_temp = comp_data_old.loc[meta_temp.index, :]
        comparison_features = comp_temp.columns
    else:
        meta_temp = meta_data_new[meta_data_new['hole_id'] == hole_id]
        comp_temp = comp_data_new.loc[meta_temp.index, :]
        comparison_features = comp_temp.columns

    """Plot Setup"""
    fig, axs = plt.subplots(
        1, comparison_plot_count + 2,
        figsize=figure_size,
    )
    fig.subplots_adjust(wspace=0.2)
    axs = axs.ravel()
    axt = plt.subplot2grid((1, comparison_plot_count + 2), (0, comparison_plot_count), colspan=2)

    axs[0].set_title(plot_name, fontsize=20, y=1.03)

    # Color list
    colors = ['#E4FA42', '#FF9C44', '#2FA0AB', '#A233B5', '#AA3939', '#2E4172', '#958E49', '#04C304',
              '#00E5E5', '#D4A6F6', '#41550F', '#FFAAAA', '#550000', '#2D882D', '#882D60']

    """Plot Comparison Columns"""
    # Loop through comparison axes
    for ax in range(len(axs) - 2):

        axs[ax].barh(
            meta_temp['depth_mean'].values,
            comp_temp[comparison_features[ax]].values,
            align='center', color=colors[ax],
            height=meta_temp['depth_length'].values, edgecolor=[0.3, 0.3, 0.3]
        )

        if ax != 0:
            axs[ax].get_yaxis().set_visible(False)
        else:
            axs[ax].set_ylabel('Depth, m', fontsize=20)

        axs[ax].locator_params(nbins=4, axis='x')
        axs[ax].tick_params(labelsize=12)
        plt.setp(axs[ax].get_xticklabels(), rotation=90)
        axs[ax].xaxis.labelpad = 10
        axs[ax].set_ylim([density_merged.dropna()['depth_from'].min(), density_merged.dropna()['depth_to'].max()])
        axs[ax].set_xlabel(comparison_features[ax] + '\nppm', fontsize=20)
        axs[ax].invert_yaxis()

    """Plot Target Column"""
    sdst_start = meta_temp['depth_from'][meta_temp['lith_dmn'] == 'SDST'].min()
    sdst_end = meta_temp['depth_to'][meta_temp['lith_dmn'] == 'SDST'].max()
    bsmt_end = meta_temp['depth_to'][meta_temp['lith_dmn'] == 'BSMT'].max()

    rect_sdst = matplotlib.patches.Rectangle((2.9, sdst_start), 0.1, (sdst_end - sdst_start),
                                             facecolor='m', edgecolor='k', lw=1, label='SDST')
    rect_bsmt = matplotlib.patches.Rectangle((2.9, sdst_end), 0.1, (bsmt_end - sdst_end),
                                             facecolor='c', edgecolor='k', lw=1, label='BSMT')

    axt.add_patch(rect_sdst)
    axt.add_patch(rect_bsmt)

    axt.plot(density['density'][density['hole_id'] == hole_id],
             density['depth'][density['hole_id'] == hole_id],
             '-', color=[0.7, 0.7, 0.7], label='Raw', lw=1)

    # axt.set_title('Training Target', fontsize=20, y=1.07)

    axt.plot(density_merged['density_filtered_median'].values, density_merged['depth_mean'].values,
             '-or', lw=2, label='Target')
    pred_index = model.dataset.hole_id_test[model.dataset.hole_id_test == plot_name].index
    axt.plot(model.y_test_pred[pred_index].values, model.dataset.depth_test.loc[pred_index, 'depth_mean'],
             '-ob', lw=2, label='Estimation')

    axt.legend(loc=1, bbox_to_anchor=(1, 1.11), frameon=False, ncol=2)

    axt.get_yaxis().set_visible(False)
    axt.locator_params(nbins=10, axis='x')
    axt.tick_params(labelsize=12)
    plt.setp(axt.get_xticklabels(), rotation=90)
    axt.xaxis.labelpad = 10
    axt.set_xlabel('Density, g/cm$^{3}$', fontsize=20)
    axt.set_xlim([2.4, 3.0])
    axt.set_ylim([density_merged.dropna()['depth_from'].min(), density_merged.dropna()['depth_to'].max()])
    axt.invert_yaxis()

    plt.plot()


def plot_depth_profiles_interactive(comp_data_old, meta_data_old, comp_data_new, meta_data_new,
                                    density, density_merged, model, figure_size=(20, 10), comparison_plot_count=5):
    # Launch widget
    interact(plot_depth_profiles,
             comp_data_old=fixed(comp_data_old),
             meta_data_old=fixed(meta_data_old),
             comp_data_new=fixed(comp_data_new),
             meta_data_new=fixed(meta_data_new),
             density=fixed(density),
             density_merged=fixed(density_merged),
             model=fixed(model),
             hole_index=(0, len(model.dataset.test_holes) - 1, 1),
             figure_size=fixed(figure_size),
             comparison_plot_count=fixed(comparison_plot_count))
