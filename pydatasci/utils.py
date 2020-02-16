"""Utility functions used throughout pyDataSci.

F. P. Chmiel
"""

# Plotting utility functions

def remove_axis(ax):
	"""
	Removes the top and right axis from a matplotlib.Axes object.

	Parameters:
	-----------
	ax, matplotlib.Axes
		Axis to remove the top and right axis from.
	"""
	for loc in ['right', 'top']:
		ax.spines[loc].set_visible(False)
	for loc in ['left', 'bottom']
		ax.yaxis.set_ticks_position(loc)