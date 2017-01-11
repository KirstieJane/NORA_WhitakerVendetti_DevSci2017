#!/usr/bin/env python

"""
This code was written by Kirstie Whitaker in January 2017 to accompany
the manuscript "Neuroscientific insights into the development of analogical reasoning".

Contact: kw401@cam.ac.uk
"""

#===============================================================================
# Import what you need
#===============================================================================
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import os
import palettable
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
import seaborn as sns
sns.set_context("poster", font_scale=1)
sns.set_style('white')
from statsmodels.formula.api import ols
import string

#===============================================================================
# Write a few useful functions
#===============================================================================

#-------------------------------------------------------------------------------
# Read in the analogy stimulus file
#-------------------------------------------------------------------------------
def read_in_analogy(analogy_stimulus_file):

    img = mpimg.imread(analogy_stimulus_file)

    img = img[80:-5, 135:-165, :]

    return img


#-------------------------------------------------------------------------------
# Read in the semantic stimulus file
#-------------------------------------------------------------------------------
def read_in_semantic(semantic_stimulus_file):

    img = mpimg.imread(semantic_stimulus_file)

    img = img[18:-55, 120:-140, :]

    return img


#-------------------------------------------------------------------------------
# Add the coloured boxes and text to the analogy example for figure 1
#-------------------------------------------------------------------------------
def add_boxes_analogy(ax, color_dict):

    color_list = [ 'orange', 'purple', 'green', 'blue' ]

    text_list = [ 'semantic\nlure', 'perceptual\nlure', 'unrelated\nlure', 'correct\nchoice']

    for i, (color, text) in enumerate(zip(color_list, text_list)):

        # Add the coloured boxes
        ax.add_patch(
            patches.Rectangle(
                (40 + i*154, 430),
                150,
                150,
                fill=False,
                edgecolor=color_dict[color],
                linewidth=3
            )
        )

        # Add the text
        ax.text(
            116 + i*154,
            623,
            text,
            fontsize=15,
            color=color_dict[color],
            horizontalalignment='center',
            verticalalignment='center',
            fontname='arial'
        )

    return ax


#-------------------------------------------------------------------------------
# Add the coloured boxes and text to the semantic example for figure 1
#-------------------------------------------------------------------------------
def add_boxes_semantic(ax, color_dict):

    color_list = [ 'red', 'green', 'green', 'purple' ]

    text_list = [ 'correct\nchoice', 'unrelated\nlure', 'unrelated\nlure', 'perceptual\nlure']

    for i, (color, text) in enumerate(zip(color_list, text_list)):

        # Add the coloured boxes
        ax.add_patch(
            patches.Rectangle(
                (51 + i*163, 455),
                160,
                160,
                fill=False,
                edgecolor=color_dict[color],
                linewidth=3
            )
        )

        # Add the text
        ax.text(
            131 + i*163,
            660,
            text,
            fontsize=15,
            color=color_dict[color],
            horizontalalignment='center',
            verticalalignment='center',
            fontname='arial'
        )

    return ax


#-------------------------------------------------------------------------------
# Add the panel labels for figure 1
#-------------------------------------------------------------------------------
def add_panel_labels_fig1(ax_list):

    coords = (0.05, 0.95)
    color='w'
    fontsize=14

    letters = string.ascii_lowercase
    for i, ax in enumerate(ax_list):

        ax.text(coords[0], coords[1],
                '({})'.format(letters[i]),
                fontsize=fontsize,
                transform=ax.transAxes,
                color=color,
                horizontalalignment='center',
                verticalalignment='center',
                fontname='arial',
                fontweight='bold'
        )

    return ax_list


#-------------------------------------------------------------------------------
# Figure out the min and max of your data and then add 5% padding
#-------------------------------------------------------------------------------
def get_min_max(data):

    data_range = np.max(data) - np.min(data)
    data_min = np.min(data) - (data_range * 0.05)
    data_max = np.max(data) + (data_range * 0.05)

    return data_min, data_max


#-------------------------------------------------------------------------------
# Add a line to your legend to make it look lovely
#-------------------------------------------------------------------------------
def add_line_to_legend(ax, color_list=['blue'], label_list=['Blue stars'], loc=0, rev=False):

    if rev:
        color_list = color_list[::-1]
        label_list = label_list[::-1]

    line_list = []
    for color, label in zip(color_list, label_list):
        line_list += [mlines.Line2D([], [], color=color, marker=None, label=label)]
    ax.legend(handles=line_list, loc=loc)

    return ax

#-------------------------------------------------------------------------------
# Report the behavioural statistical models with age and behaviour
#-------------------------------------------------------------------------------
def report_behav_age_correlations(y_name, df):

    formula = '{} ~ Age_scan'.format(y_name)
    mod = ols(formula=formula, data=df)
    res_lin = mod.fit()

    formula = '{} ~ Age_scan_sq + Age_scan'.format(y_name)
    mod = ols(formula=formula, data=df)
    res_quad = mod.fit()

    if 'R2' in y_name and not 'dis' in y_name and not 'sem' in y_name and not y_name.endswith('per'):
        formula = '{} ~ Age_scan_sq + Age_scan + {}'.format(y_name, y_name.replace('R2', 'R1'))
        mod = ols(formula=formula, data=df)
        res_corr = mod.fit()

    if 'R1' in y_name and not 'dis' in y_name and not 'sem' in y_name and not y_name.endswith('per'):
        formula = '{} ~ Age_scan_sq + Age_scan + {}'.format(y_name, y_name.replace('R1', 'R2'))
        mod = ols(formula=formula, data=df)
        res_corr = mod.fit()

    print ('=== {} ==='.format(y_name))
    print ('Linear w age')
    print ('  Beta(Age) = {:2.4f}, P = {:2.4f}'.format(res_lin.params['Age_scan'], res_lin.pvalues['Age_scan']))
    print ('  Rsq = {:2.3f}, Rsq_adj = {:2.3f}'.format(res_lin.rsquared, res_lin.rsquared_adj))
    print ('  F({}, {}) = {:2.3f}, P = {:2.4f}'.format(res_lin.df_model, res_lin.df_resid, res_lin.fvalue, res_lin.f_pvalue))
    print ('Quadratic w age')
    print ('  Beta(AgeSq) = {:2.4f}, P = {:2.4f}'.format(res_quad.params['Age_scan_sq'], res_quad.pvalues['Age_scan_sq']))
    print ('  Beta(Age) = {:2.4f}, P = {:2.4f}'.format(res_quad.params['Age_scan'], res_quad.pvalues['Age_scan']))
    print ('  Rsq = {:2.3f}, Rsq_adj = {:2.3f}'.format(res_quad.rsquared, res_quad.rsquared_adj))
    print ('  F({}, {}) = {:2.3f}, P = {:2.4f}'.format(res_quad.df_model, res_quad.df_resid, res_quad.fvalue, res_quad.f_pvalue))
    if 'R2' in y_name and not 'dis' in y_name and not 'sem' in y_name and not y_name.endswith('per'):
        print ('Quadratic w age correcting for accuracy')
        print ('  Beta(R1) = {:2.4f}, P = {:2.4f}'.format(res_corr.params[y_name.replace('R2', 'R1')],
                                                       res_corr.pvalues[y_name.replace('R2', 'R1')]))
        print ('  Beta(AgeSq) = {:2.4f}, P = {:2.4f}'.format(res_corr.params['Age_scan_sq'], res_corr.pvalues['Age_scan_sq']))
        print ('  Beta(Age) = {:2.4f}, P = {:2.4f}'.format(res_corr.params['Age_scan'], res_corr.pvalues['Age_scan']))
        print ('  Rsq = {:2.3f}, Rsq_adj = {:2.3f}'.format(res_corr.rsquared, res_corr.rsquared_adj))
        print ('  F({}, {}) = {:2.3f}, P = {:2.4f}'.format(res_corr.df_model, res_corr.df_resid, res_corr.fvalue, res_corr.f_pvalue))
    if 'R1' in y_name and not 'dis' in y_name and not 'sem' in y_name and not y_name.endswith('per'):
        print ('Quadratic w age correcting for accuracy')
        print ('  Beta(R2) = {:2.4f}, P = {:2.4f}'.format(res_corr.params[y_name.replace('R1', 'R2')],
                                                       res_corr.pvalues[y_name.replace('R1', 'R2')]))
        print ('  Beta(AgeSq) = {:2.4f}, P = {:2.4f}'.format(res_corr.params['Age_scan_sq'], res_corr.pvalues['Age_scan_sq']))
        print ('  Beta(Age) = {:2.4f}, P = {:2.4f}'.format(res_corr.params['Age_scan'], res_corr.pvalues['Age_scan']))
        print ('  Rsq = {:2.3f}, Rsq_adj = {:2.3f}'.format(res_corr.rsquared, res_corr.rsquared_adj))
        print ('  F({}, {}) = {:2.3f}, P = {:2.4f}'.format(res_corr.df_model, res_corr.df_resid, res_corr.fvalue, res_corr.f_pvalue))


#-------------------------------------------------------------------------------
# Add the panel labels to figure 2
#-------------------------------------------------------------------------------
def add_panel_labels_fig2(ax_list):

    x_list = [ -0.175, -0.115, -0.145 ]
    y = 1.0
    color='k'
    fontsize=18

    letters = string.ascii_lowercase
    for i, ax in enumerate(ax_list):

        ax.text(x_list[i], y,
                '({})'.format(letters[i]),
                fontsize=fontsize,
                transform=ax.transAxes,
                color=color,
                horizontalalignment='center',
                verticalalignment='center',
                fontname='arial',
                fontweight='bold'
        )

    return ax_list


#-------------------------------------------------------------------------------
# Read in the pial surface brain images
#-------------------------------------------------------------------------------
def read_in_brains(results_surface_file):

    img = mpimg.imread(results_surface_file)

    img = img[21:-120, 44:-44, :]

    return img


#-------------------------------------------------------------------------------
# Read in the venn diagram
#-------------------------------------------------------------------------------
def read_in_venn(venn_file):

    img = mpimg.imread(venn_file)

    return img


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def report_cluster_stats(f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4, cluster=1):

    name_dict = { 1 : 'SEMANTIC > FIX',
                  2 : 'ANALOGY > FIX',
                  4 : 'ANA > SEM' }
    file_dict = { 1 : f_mri_cope1,
                  2 : f_mri_cope2,
                  4 : f_mri_cope4 }

    for cope, name in name_dict.items():
        print('=== {} ==='.format(name))
        df = read_in_data(f_behav, file_dict[cope])

        print('  Corr cluster {} w age'.format(cluster))
        x_name = 'Age_scan'
        y_name = 'cluster_{}'.format(cluster)
        report_correlation(df, x_name, y_name)

        print('  Corr cluster 1 w R2acc')
        x_name = 'R2_percent_acc'
        y_name = 'cluster_{}'.format(cluster)
        report_correlation(df, x_name, y_name)

        print('  Corr cluster 1 w R2acc covar age')
        x_name = 'R2_percent_acc'
        y_name = 'cluster_{}'.format(cluster)
        covar_name = 'Age_scan'
        report_correlation(df, x_name, y_name, covar_name=covar_name)


#-------------------------------------------------------------------------------
#  Report a correlation (partial if covariates are provided)
#-------------------------------------------------------------------------------
def report_correlation(df, x_name, y_name, covar_name=None):

    if not covar_name:
        r, p = pearsonr(df[x_name], df[y_name])

    else:
        x_res = residuals(df[covar_name], df[x_name])
        y_res = residuals(df[covar_name], df[y_name])

        df['{}_res'.format(x_name)] = x_res
        df['{}_res'.format(y_name)] = y_res

        r, p = pearsonr(df['{}_res'.format(x_name)], df['{}_res'.format(y_name)])

    # Format nicely
    r, p = format_r_p(r, p, r_dp=3)

    print('    r {}, p {}'.format(r, p))


#-------------------------------------------------------------------------------
# Format r and p values to print out nicely
#-------------------------------------------------------------------------------
def format_r_p(r, p, r_dp=2):

    r = '{:2.{width}f}'.format(r, width=r_dp)
    r = '= {}'.format(r)

    if p < 0.001:
        p = '< .001'
    else:
        p = '{:2.3f}'.format(p)
        p = '= {}'.format(p[1:])

    return r, p


#-------------------------------------------------------------------------------
# Read in the behavioural and extracted regional MRI values
#-------------------------------------------------------------------------------
def read_in_data(f_behav, f_mri):
    behav_df = pd.read_csv(f_behav)

    mri_df = pd.read_csv(f_mri, sep=r"\s*", engine='python')
    mri_df['subid_long'] = mri_df['sub_id']

    df = behav_df.merge(mri_df, on='subid_long')

    df.loc[:, 'Age_scan_sq'] = df.loc[:, 'Age_scan']**2

    return df

#-------------------------------------------------------------------------------
# Calculate residuals for a given covariate
#-------------------------------------------------------------------------------
def residuals(x, y):
    '''
    A useful little function that correlates
    x and y together to give their residual
    values. These can then be used to calculate
    partial correlation values
    '''
    import numpy as np

    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    A = np.vstack([x, np.ones(x.shape[-1])]).T
    B = np.linalg.lstsq(A, y)[0]
    m = B[:-1]
    c = B[-1]
    pre = np.sum(m * x.T, axis=1) + c
    res = y - pre
    return res


#-------------------------------------------------------------------------------
# Read in just the left lateral brain image
#-------------------------------------------------------------------------------
def read_in_leftlatbrain(results_surface_file):

    img = mpimg.imread(results_surface_file)

    img = img[25:-665, 35:-785, :]

    return img

#-------------------------------------------------------------------------------
# Add a circle to figure 5
#-------------------------------------------------------------------------------
def add_circle(ax):

    circle = plt.Circle((150, 350), 90,
                        linestyle='dashed',
                        fill=False,
                        edgecolor='k',
                        linewidth=4)
    ax.add_artist(circle)

    return ax

#-------------------------------------------------------------------------------
# Make the scatter plots for figure 5
#-------------------------------------------------------------------------------
def figure5_scatterplots(f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4, ax_list, cluster=1, show_r_p=False):
    '''
    Create a 2x2 grid of scatter plots.

    [0,0] - sem > baseline vs age
    [0,1] - sem > baseline vs R2_accuracy PARTIAL
    [0,1] - ana > baseline vs age
    [1,1] - ana > baseline vs R2_accuracy PARTIAL

    '''
    df_cope1 = read_in_data(f_behav, f_mri_cope1)
    df_cope2 = read_in_data(f_behav, f_mri_cope2)

    colors_dict = { 'Age_scan' : sns.color_palette()[2],
                    'R2_percent_acc' : sns.color_palette()[1] }

    ax_list = ax_list.reshape(-1)

    x_name_list = [ 'Age_scan', 'R2_percent_acc' ]
    x_label_dict = { 'Age_scan' : 'Age (years)',
                     'Age_scan_res' : 'Age (years) [Partial]',
                     'R2_percent_acc' : 'Analogy accuracy (%)',
                     'R2_percent_acc_res' : 'Analogy accuracy (%) [Partial]',
                     'R2_percent_sem' : 'Semantic errors (%)',
                     'R2_Correct_dividedby_Semantic' : 'Accuracy / Semantic Err' }
    y_label_dict = { 0 : 'Semantic > Baseline',
                     1 : 'Analogy > Baseline' }

    df_dict = { 0 : df_cope1,
                1 : df_cope2 }

    for i, ax in enumerate(ax_list[::2]):

        x_name = 'Age_scan'
        y_name = 'cluster_{}'.format(cluster)

        df = df_dict[i]

        sns.regplot(x_name, y_name, data=df, ax=ax, color=colors_dict[x_name])

        ax.locator_params(nbins=4)

        ax.set_xlim(get_min_max(df[x_name]))
        ax.set_ylim(get_min_max(df[y_name]))
        ax.set_xlabel(x_label_dict[x_name])
        ax.set_ylabel(y_label_dict[i])

        r, p = pearsonr(df[x_name], df[y_name])
        r, p = format_r_p(r, p)

        ax.axhline(0, color='k', linestyle='dashed', linewidth=1)

        if show_r_p:
            ax.text(0.05, 0.95,
                    'r {}\np {}'.format(r, p),
                    transform=ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    size='large')

        if i == 0:
            ax.set_xlabel('')
        ax.yaxis.set_label_coords(-0.18, 0.5)

    for i, ax in enumerate(ax_list[1::2]):

        x_name = 'R2_percent_acc'
        y_name = 'cluster_{}'.format(cluster)
        covar_name = 'Age_scan'

        df = df_dict[i]

        x_res = residuals(df[covar_name], df[x_name])
        y_res = residuals(df[covar_name], df[y_name])

        df['{}_res'.format(x_name)] = x_res
        df['{}_res'.format(y_name)] = y_res

        sns.regplot('{}_res'.format(x_name),
                    '{}_res'.format(y_name),
                    data=df,
                    ax=ax,
                    color=colors_dict[x_name])

        ax.locator_params(nbins=4)

        ax.set_xlim(get_min_max(df['{}_res'.format(x_name)]))
        ax.set_ylim(get_min_max(df['{}_res'.format(y_name)]))
        ax.set_xlabel('{} [Partial]'.format(x_label_dict[x_name]))
        ax.set_ylabel('{} [Partial]'.format(y_label_dict[i]))

        r, p = pearsonr(df['{}_res'.format(x_name)], df['{}_res'.format(y_name)])
        r, p = format_r_p(r, p)

        ax.axhline(0, color='k', linestyle='dashed', linewidth=1)

        if show_r_p:
            ax.text(0.05, 0.95,
                    'r {}\np {}'.format(r, p),
                    transform=ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    size='large')

        if i == 0:
            ax.set_xlabel('')

        ax.yaxis.set_label_coords(-0.18, 0.5)

    sns.despine()

    return ax_list


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Add panel labels to figure 5
#-------------------------------------------------------------------------------
def add_panel_labels_fig5(ax_brain, ax_list):

    # First the letters
    letter_ax_list = [ax_brain, ax_list[0], ax_list[2]]
    x_list = [ 0.1, -0.24, -0.24 ]
    y_list = [ 0.95, 1.0, 1.0 ]
    color='k'
    fontsize=18

    letters = string.ascii_lowercase
    for i, ax in enumerate(letter_ax_list):

        ax.text(x_list[i], y_list[i],
                '({})'.format(letters[i]),
                fontsize=fontsize,
                transform=ax.transAxes,
                color=color,
                horizontalalignment='center',
                verticalalignment='center',
                fontname='arial',
                fontweight='bold'
        )

    # Then the lowercase roman numerals
    color='k'
    fontsize=18

    numerals = [ 'i', 'ii', 'i', 'ii' ]

    for i, ax in enumerate(ax_list):

        ax.text(0.02, 0.95,
                '{}'.format(numerals[i]),
                fontsize=fontsize,
                transform=ax.transAxes,
                color=color,
                horizontalalignment='left',
                verticalalignment='center',
                fontname='arial',
                fontweight='bold'
        )
    return ax_brain, ax_list



#===============================================================================
# Now write your main figure & stats reporting functions
# Note that some of the stats reporting functions are above, these ones below
# the line are likely to be called directly from the jupyter notebook.
#===============================================================================

#-------------------------------------------------------------------------------
# Figure 1
#-------------------------------------------------------------------------------
def make_figure1(analogy_stimulus_file, semantic_stimulus_file, color_dict):

    fig, ax_list = plt.subplots(1, 2, figsize=(9,4.5))

    # Put the example analogy stimulus on the left
    ax = ax_list[0]
    img_ana = read_in_analogy(analogy_stimulus_file)
    ax.imshow(img_ana)

    # Add the coloured boxes & text
    ax = add_boxes_analogy(ax, color_dict)

    # Put the example semantic stimulus on the right
    ax = ax_list[1]
    img_sem = read_in_semantic(semantic_stimulus_file)
    ax.imshow(img_sem)

    # Add the coloured boxes & text
    ax = add_boxes_semantic(ax, color_dict)

    # Add in panel labels
    ax_list = add_panel_labels_fig1(ax_list)

    # Turn off both of the axes
    for ax in ax_list:
        ax.set_axis_off()

    # Tighten up the layout
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)

    fig.savefig('../FIGURES/Figure1.png', dpi=600, bbox_inches=0)
    fig.savefig('../FIGURES/Figure1.pdf', dpi=600, bbox_inches=0)

    plt.show()

#-------------------------------------------------------------------------------
# Figure 2
#-------------------------------------------------------------------------------
def make_figure2(f_behav):

    df = pd.read_csv(f_behav)

    df.loc[:, 'Age_scan_sq'] = df.loc[:,'Age_scan']**2

    # Define the color list
    color_list = palettable.colorbrewer.get_map('Set1', 'qualitative', 5).mpl_colors

    colors_dict = { 'R1_percent_acc' : color_list[0],
                    'R2_percent_acc' : color_list[1],
                    'R1_meanRTcorr_cor' : color_list[0],
                    'R2_meanRTcorr_cor' : color_list[1],
                    'R2_percent_dis' : color_list[2],
                    'R2_percent_per' : color_list[3],
                    'R2_percent_sem' : color_list[4] }

    fig, ax_list = plt.subplots(1,3, figsize=(16,4.5))

    ax_list = ax_list.reshape(-1)

    x_label_dict = { 'Age_scan' : 'Age (years)' }
    y_label_dict = { 0 : 'Accuracy (% resp)',
                     1 : 'Reaction time (s)',
                     2 : 'Analogy error rate (% resp)' }
    y_measures_dict = { 0 : ['R1_percent_acc', 'R2_percent_acc'],
                        1 : ['R1_meanRTcorr_cor', 'R2_meanRTcorr_cor'],
                        2 : ['R2_percent_dis', 'R2_percent_per', 'R2_percent_sem']}
    y_measures_label_dict = { 'R1_percent_acc' : 'Semantic',
                              'R2_percent_acc' : 'Analogy',
                              'R1_meanRTcorr_cor' : 'Semantic',
                              'R2_meanRTcorr_cor' : 'Analogy',
                              'R2_percent_dis' : 'Unrelated',
                              'R2_percent_per' : 'Perceptual',
                              'R2_percent_sem' : 'Semantic' }
    legend_loc_dict = { 0 : 4,
                        1 : 1,
                        2 : 1 }
    legend_rev_dict = { 0 : False,
                        1 : False,
                        2 : True }

    for i, ax in enumerate(ax_list):

        x_name = 'Age_scan'
        y_name_list = y_measures_dict[i]

        colors_list = []
        labels_list = []

        for y_name in y_name_list:
            sns.regplot(x_name, y_name, data=df,
                        ax=ax,
                        color=colors_dict[y_name],
                        order=2)
            colors_list += [colors_dict[y_name]]
            labels_list += [y_measures_label_dict[y_name]]

        ax.locator_params(nbins=6, axis='y')
        ax.set_xticks([6, 10, 14, 18])

        ax.set_xlim(get_min_max(df[x_name]))
        ax.set_ylim(get_min_max(df[y_name]))
        ax.set_xlabel(x_label_dict[x_name])
        ax.set_ylabel(y_label_dict[i])

        add_line_to_legend(ax,
                           color_list=colors_list,
                           label_list=labels_list,
                           loc=legend_loc_dict[i],
                           rev=legend_rev_dict[i])

    sns.despine()

    # Add in panel labels
    ax_list = add_panel_labels_fig2(ax_list)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    fig.savefig('../FIGURES/Figure2.png', dpi=600, bbox_inches=0)
    fig.savefig('../FIGURES/Figure2.pdf', dpi=600, bbox_inches=0)

    plt.show()

#-------------------------------------------------------------------------------
# Report the descriptive behavoural stats
#-------------------------------------------------------------------------------
def report_behav_stats(f_behav):

    df = pd.read_csv(f_behav)
    df.loc[:, 'Age_scan_sq'] = df.loc[:,'Age_scan']**2

    print('===== Behavioural Statistics ======')
    for measure in ['R1_percent_acc', 'R2_percent_acc',
                    'R1_meanRTcorr_cor', 'R2_meanRTcorr_cor',
                    'R2_percent_sem', 'R2_percent_per', 'R2_percent_dis']:
        print('{}: N = {:2.0f}, M = {:2.3f}, SD = {:2.3f}'.format(measure, df[measure].notnull().count(), df[measure].mean(), df[measure].std()))

        if 'R2' in measure and not 'dis' in measure and not 'sem' in measure and not measure.endswith('per'):
            measure_diff = measure.replace('R2', 'R2_sub_R1')
            df[measure_diff] = df[measure] - df[measure.replace('R2', 'R1')]
            print('{}: M = {:2.3f}, SD = {:2.3f}'.format(measure_diff, df[measure_diff].mean(), df[measure_diff].std()))
            print('    N R2 gt R1 = {}, N R2 lt R1 = {}, N same = {}'.format(np.sum(df[measure_diff]>0),
                                                                         np.sum(df[measure_diff]<0),
                                                                         np.sum(df[measure_diff]==0)))

        if 'R1' in measure:
            t, p = ttest_rel(df[measure], df[measure.replace('R1', 'R2')])
            print('   R1 vs R2 (paired): t({:2.0f}) = {:2.3f}, p = {:2.3f}'.format(df[measure].count()-1, t, p))

        if 'sem' in measure:
            t, p = ttest_rel(df['R2_percent_sem'], df['R2_percent_per'])
            print('   sem vs per (paired): t({:2.0f}) = {:2.3f}, p = {:2.3f}'.format(df['R2_percent_per'].count()-1, t, p))
        if 'dis' in measure:
            t, p = ttest_rel(df['R2_percent_per'], df['R2_percent_dis'])
            print('   per vs dis (paired): t({:2.0f}) = {:2.3f}, p = {:2.3f}'.format(df['R2_percent_per'].count()-1, t, p))

    print('\n===== Correlations with age =====')
    measures_list = ['R1_percent_acc', 'R2_percent_acc',
                     'R1_meanRTcorr_cor', 'R2_meanRTcorr_cor',
                     'R2_percent_dis', 'R2_percent_per', 'R2_percent_sem']

    for measure in measures_list:
        report_behav_age_correlations(measure, df)


#-------------------------------------------------------------------------------
# Figure 3
#-------------------------------------------------------------------------------
def make_figure3(mean_results_surface_file, venn_file):

    fig, ax = plt.subplots(figsize=(6,4.5))

    # Add in the brains
    img_brains = read_in_brains(mean_results_surface_file)

    ax.imshow(img_brains)

    # Turn off the axes
    ax.set_axis_off()

    # Overlay the venn diagram
    ax_venn = fig.add_axes([0.25, 0.21, 0.5, 0.5]) # inset axes
    img_venn = read_in_venn(venn_file)

    ax_venn.imshow(img_venn)

    # Turn off the axes
    ax_venn.set_axis_off()

    # Tighten up the layout
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)

    fig.savefig('../FIGURES/Figure3.png', dpi=600, bbox_inches=0)
    fig.savefig('../FIGURES/Figure3.pdf', dpi=600, bbox_inches=0)

    plt.show()

#-------------------------------------------------------------------------------
# Figure 4
#-------------------------------------------------------------------------------
def make_figure4(corrage_results_surface_file, venn_file, f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4):

    # Report the cluster statistics
    report_cluster_stats(f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4, cluster=1)

    # Now make the figure
    fig, ax = plt.subplots(figsize=(6,4.5))

    # Add in the brains
    img_brains = read_in_brains(corrage_results_surface_file)

    ax.imshow(img_brains)

    # Turn off the axes
    ax.set_axis_off()

    # Overlay the venn diagram
    ax_venn = fig.add_axes([0.25, 0.21, 0.5, 0.5]) # Add in a new axis
    img_venn = read_in_venn(venn_file)

    ax_venn.imshow(img_venn)

    # Turn off the axes
    ax_venn.set_axis_off()

    # Tighten up the layout
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)

    fig.savefig('../FIGURES/Figure4.png', dpi=600, bbox_inches=0)
    fig.savefig('../FIGURES/Figure4.pdf', dpi=600, bbox_inches=0)

    plt.show()

#-------------------------------------------------------------------------------
# Figure 5
#-------------------------------------------------------------------------------
def make_figure5(corracc_results_surface_file, f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4):

    # Report the cluster statistics
    report_cluster_stats(f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4, cluster=1)

    fig, ax_list = plt.subplots(2,2, figsize=(14,7.5))

    # Add in left lateral brain
    ax_brain = fig.add_axes([-0.01, 0.2, 0.36, 0.6]) # Add in the brain axis

    img_brain = read_in_leftlatbrain(corracc_results_surface_file)

    ax_brain.imshow(img_brain)

    # Add the dashed circle
    ax_brain = add_circle(ax_brain)

    # Turn off the axes
    ax_brain.set_axis_off()

    # Shift these subplots over to the right
    # to make space for the brain
    fig.subplots_adjust(left=0.42, right=0.99, bottom=0.1, top=0.97, wspace=0.3)

    # Add in the scatter plots
    ax_list = figure5_scatterplots(f_behav, f_mri_cope1, f_mri_cope2, f_mri_cope4,
                                   ax_list, cluster=1, show_r_p=False)

    # Add in the labels
    ax_brain, ax_list = add_panel_labels_fig5(ax_brain, ax_list)

    # Save the figure
    fig.savefig('../FIGURES/Figure5.png', dpi=600, bbox_inches=0)
    fig.savefig('../FIGURES/Figure5.pdf', dpi=600, bbox_inches=0)

    plt.show()
