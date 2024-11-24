"""
This script contains a set of functions to customize matplotlib plots. 
These functions allow users to easily modify aspects of a plot such as titles, 
labels, spines, ticks, and legends for improved visual appeal and clarity.

This module should be imported and contains the following:
  * set_titles_and_labels: Sets the titles and labels for the axes.
  * customize_spines: Customizes the appearance of the subplot's spines.
  * customize_ticks: Customizes the tick marks and labels on the axes.
  
"""

import matplotlib
import numpy as np
from typing import Dict, Union

# Define default styling parameters
DEFAULT_FONT_SIZE = 14
DEFAULT_LINE_WIDTH = 2.5
DEFAULT_COLOR = '0.2'
DEFAULT_FONT_WEIGHT = 'bold'
DEFAULT_SPINE_COLOR = {'bottom': DEFAULT_COLOR, 'left': DEFAULT_COLOR}
DEFAULT_TICK_PARAMS = {'axis': 'both', 'width': DEFAULT_LINE_WIDTH}
DEFAULT_LABEL_FONTDICT = {
    'weight': 'bold', 'fontsize': DEFAULT_FONT_SIZE, 'color': DEFAULT_COLOR
}
DEFAULT_LEGEND_PROP = {"size": DEFAULT_FONT_SIZE}
DEFAULT_LEGEND_LINE_WIDTH = 3.5


def set_titles_and_labels(
    ax: matplotlib.axes.Axes, title: str = None, xlabel: str = None,
    ylabel: str = None,
    fontdict: Dict[str, Union[str, float]] = DEFAULT_LABEL_FONTDICT
):
  """
  Set titles and labels for the axes.

  Args:
    ax (matplotlib.axes.Axes): The axes object of the subplot.
    title (str, optional): The title of the subplot. Defaults to None.
    xlabel (str, optional): The label for the x-axis. Defaults to None.
    ylabel (str, optional): The label for the y-axis. Defaults to None.
    fontdict (Dict[str, Union[str, float]], optional): Dictionary for font
        properties. Defaults to DEFAULT_LABEL_FONTDICT.

  """
  if title:
    ax.set_title(title, **fontdict)
  if xlabel:
    ax.set_xlabel(xlabel, **fontdict)
  if ylabel:
    ax.set_ylabel(ylabel, **fontdict)


def customize_spines(
    ax: matplotlib.axes.Axes, spine_colors: Dict[str,
                                                 str] = DEFAULT_SPINE_COLOR,
    linewidth: float = DEFAULT_LINE_WIDTH
):
  """
  Customize the appearance of the subplot's spines.

  Args:
    ax (matplotlib.axes.Axes): The axes object of the subplot.
    spine_colors (Dict[str, str], optional): Dictionary specifying colors for
        each visible spine. Defaults to DEFAULT_SPINE_COLOR.
    linewidth (float, optional): Width of the spines. Defaults to
        DEFAULT_LINE_WIDTH.

  """
  for spine in spine_colors:
    ax.spines[spine].set_linewidth(linewidth)
    ax.spines[spine].set_color(spine_colors[spine])
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


def customize_ticks(
    ax: matplotlib.axes.Axes, remove_ticks_lines: bool = False,
    rotate_x_ticks: int = 0,
    tick_params: Dict[str, Union[str, float]] = DEFAULT_TICK_PARAMS,
    label_fontdict: Dict[str, Union[str, float]] = DEFAULT_LABEL_FONTDICT
):
  """
  Customize the tick marks and labels.

  Args:
    ax (matplotlib.axes.Axes): The axes object of the subplot.
    tick_params (Dict[str, Union[str, float]], optional): Parameters for tick 
        customization. Defaults to DEFAULT_TICK_PARAMS.
    label_fontdict (Dict[str, Union[str, float]], optional): Font properties
        for tick labels.  Defaults to DEFAULT_LABEL_FONTDICT. 
  """
  if tick_params:
    ax.tick_params(**tick_params)
  if rotate_x_ticks:
    for label in ax.get_xticklabels():
      label.set_rotation(rotate_x_ticks)
  if label_fontdict:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
      label.set_fontweight(label_fontdict.get('weight'))
      label.set_fontsize(label_fontdict.get('fontsize'))
      label.set_color(label_fontdict.get('color'))
  if remove_ticks_lines:
    ax.tick_params(axis='both', length=0)


def customize_colorbar(
    cbar: matplotlib.colorbar.Colorbar, number_of_ticks: int = 5,
    tick_params: Dict[str, Union[str, float]] = DEFAULT_TICK_PARAMS,
    label_fontdict: Dict[str, Union[str, float]] = DEFAULT_LABEL_FONTDICT
):
  """
  Customize the color bar in the plot.

  Args:
    cbar (matplotlib.colorbar.Colorbar): The color bar object of the subplot.
    number_of_ticks (int): Number of ticks in the color bar including min, max.
    tick_params (Dict[str, Union[str, float]], optional): Parameters for tick
        customization.
    label_fontdict (Dict[str, Union[str, float]], optional): Font properties
        for tick labels.

  """

  vmin, vmax = cbar.mappable.get_clim()
  vmin_rounded = np.floor(vmin)
  vmax_rounded = np.ceil(vmax)
  ticks = np.linspace(vmin_rounded, vmax_rounded, number_of_ticks)
  ticks = np.unique(np.concatenate(([vmin_rounded], ticks, [vmax_rounded])))
  cbar.set_ticks(ticks)
  cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
  cbar.mappable.set_clim(vmin_rounded, vmax_rounded)

  cbar.ax.tick_params(**tick_params)
  for label in cbar.ax.get_yticklabels():
    label.set_fontweight(label_fontdict.get('weight'))
    label.set_fontsize(label_fontdict.get('fontsize'))
    label.set_color(label_fontdict.get('color'))

  for spine in cbar.ax.spines.values():
    spine.set_linewidth(DEFAULT_LINE_WIDTH)
    spine.set_color(DEFAULT_COLOR)
