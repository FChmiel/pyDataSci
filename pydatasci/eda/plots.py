"""Plots used throughout the exploratory analysis pipeline.

F. P. Chmiel"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# parameters for text boxes
TBOX_PARAMS = {'facecolor':'white', 'alpha':0.75, 'linewidth':0.75,
			   'edgecolor':'gray', 'boxstyle':'round,pad=0.5'}


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

def binomial_distribution(df, feature, target):
	"""
	To be created.
	"""
	pass