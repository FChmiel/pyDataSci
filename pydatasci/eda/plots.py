"""
Plots summarizing feature importance

pyDataSci, Helper functions for binary classification problems.

Copyright (C) 2020  F. P. Chmiel

Email : francispeterchmiel@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

import utils

# parameters for text boxes
TBOX_PARAMS = {'facecolor':'white',
			   'alpha':0.75,
			   'linewidth':0.75,
			   'edgecolor':'gray',
			   'boxstyle':'round,pad=0.5'}


def continuous_distribution(df, feature, target):
	"""
	Plots a distribution for a continuous feature, stratified by the binary 
	target. A Kolomogorov-Smirnov two-sample test is performed to check if 
	the two samples are drawn from the same distribution.

	A small D, or high p means we cannot reject the Null hypothesis that the
	two distributions are drawn from the same distribution.

	Parameters:
	----------
	df, pd.DataFrame
		Pandas dataframe containing the data to plot.

	feature, str
		Name of column in df containing continuous variable to plot.

	target, {str, array-like}
		Name of the column containing the target or array containing the 
		target for each instance in df. The target must be binary.

	Returns:
	--------
	D, float
		The 2-sample KS statistic

	p, float 
		p-value for the 2-sample KS statistic. Tests the Null hypothesis that
		the two distributions are drawn from the same distribution.
	
	fig, matplotlib.figure.Figure
		Figure instance the distribution was plotted too.
	"""

	if type(target)==str:
		target = df["target"].values
	# check the target is binary
	if not np.array_equal(target, target.astype(bool)):
		raise ValueError("target must be binary.")

	# create target mask
	pos_mask = target==1

	# test if samples were drawn from same distributions.
	D, p = ks_2samp(df.loc[pos_mask, feature], df.loc[~pos_mask, feature])

	# plot the distributions
	fig, ax = plt.subplots()
	masks = [pos_mask, ~pos_mask]
	colors = ['r', 'b']
	for m, c in zip(masks, colors):	
		sns.kdeplot(df.loc[m, feature], ax=ax, shade=True, color=c)

	# add text box describing distributions
	ks_desc = "$D$ : {0:.3f}\n$p$ : {1:.3f}".format(D, p)
	ax.text(0.66, 0.67, ks_desc, transform=ax.transAxes, bbox=TBOX_PARAMS)
	ax.text(0.64, 0.58, 'KS 2-sample', transform=ax.transAxes, alpha=0.9)

	# format the axes
	ax.set_ylabel('Probability density')
	ax.set_xlabel(feature)

	return D, p, fig

def binomial_distribution(df,
                          feature,
                          target,
                          ylabel='Mean target',
                          ci_method='wilson', 
                          quiet=False):
    """
    Plots a binary distribution where the y-axis is the mean target 
    for each category. The feature must have 2 categories only. A significance
    test is performed, testing the null hypothesis that the distribution of
    targets across each category are drawn from the same distribution.

    Paramters:
    ----------
    df : pd.DataFrame,
        Dataframe containing data to plot.

    feature : str,
        Name of the column in df to plot.
    
    target : {str, array-like},
        Must be name of column in df containing the binary target or numpy 
		array containing binary target for each instance in df.

    ylabel : str (default='Mean target'),
        Text to label the y-axis with.
        
    ci_method : (default=wilson),
        Passed to statsmodels.stats.proportion.proportion_confint, method used 
		to calculate 95 % confidence intervals.
    
    quiet : bool (default=False),
        Whether to print calculated statistics to screen.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure,
        Figure object.
    
    ax : matplotlib.pyplot.axis,
        Axis plotted to.
    """
    if type(target)!=str:
        df['target'] = target
        target = 'target'

    # check feature is binary
    if df[feature].nunique()!=2:
        raise Exception('Feature must be binary.')
        
    # check target is binary
    if df[target].nunique()!=2:
        raise Exception('Target must be binary.')

    agg_df = df.groupby(feature)['target'].agg(['mean', 'count', 'sum'])
    
    # Calculate confidence intervals and p-value
    nocc = agg_df['sum'] # number of occurences of event
    nobs = agg_df['count'] # number of observations
    ci_lower, ci_upper = proportion_confint(nocc, nobs, method=ci_method)
    z, p = proportions_ztest(nocc, nobs, alternative='two-sided')
    
    if not quiet:
        print('----'*5)
        print('Variable:', feature)
        print('----'*5)
        print(agg_df)
        print('')
        print(f'Two-sided z score: {z:.4f}')
        print(f'p-value: {p:.4f}')
        print('')
        print('ci_lower\n', ci_lower)
        print('')
        print('ci_upper\n', ci_upper)
    
    # create the plot
    fig, ax = plt.subplots()
    yerr = np.vstack([agg_df['mean']-ci_lower, ci_upper-agg_df['mean']])
    ax.bar(agg_df.index,
           agg_df['mean'],
           yerr=yerr.reshape((-1,2)),
           color='grey',
           capsize=2)
    
    # format the plot
    ax.set_title(f'{feature.capitalize()}, $p$: {p:.3f}', fontsize=10)
    ax.tick_params(which='both', labelsize=8)
    utils.remove_axis(ax)
    ax.set_ylabel(ylabel, fontsize=9)
    fig.set_size_inches(3.5,2.5)
    return fig, ax