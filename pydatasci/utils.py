"""
Utility functions used throughout pyDataSci

pyDataSci, Helper functions for binary classification problems.

Copyright (C) 2020  F. P. Chmiel

Email:francischmiel@hotmail.co.uk

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