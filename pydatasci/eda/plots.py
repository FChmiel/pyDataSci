"""Plots used throughout the exploratory analysis pipeline.

F. P. Chmiel"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


def continuous_distribution(df, feature, target):
	"""
	Plots a distribution for a continuous feature, stratified by the binary 
	target. A Kolomogorov-Smirnov two-sample test is performed to check if 
	the two samples are drawn from the same distribution.

	Parameters:
	----------
	df, pd.DataFrame
		Pandas dataframe

	target, {str, array-like}
		Name of the column containing the target or array containing the 
		target for each instance in df. The target must be binary.

	Returns:
	--------
	fig, matplotlib.figure.Figure
		The figure the distribution was plotted too.
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


def binomial_distribution(df, feature, target):
	"""
	To be created.
	"""
	pass