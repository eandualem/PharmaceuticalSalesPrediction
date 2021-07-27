import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class VisSeaborn():

  def __init__(self):
    pass

  def single_plot(self, plot, title, x_labels, y_labels, figsize):
      fig, ax = plt.subplots(figsize=figsize)
      _ax = lambda __, _: ax
      plot(0, 0, _ax)
      plt.title(title, fontsize=15, fontweight='bold')
      plt.xlabel(x_labels[0], fontsize=14)
      plt.ylabel(y_labels[0], fontsize=14)
      fig.show()

  def vertical_plot(self, plot, rows, title, subtitles, x_labels, y_labels, figsize):
    fig, ax = plt.subplots(rows, 1, sharex=True, figsize=figsize)
    _ax = lambda x, _: ax[x]
    for r in range(rows):
      ax[r].set_title(subtitles[r])
      ax[r].set(xlabel=x_labels[r], ylabel='')
      plot(r, 0, _ax)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.subplots_adjust(hspace=0.2)
    fig.show()
  
  def horizontal_plot(self, plot, cols, title, subtitles, x_labels, y_labels, figsize):
    fig, ax = plt.subplots(1, cols, sharex=True, figsize=figsize)
    _ax = lambda _, y: ax[y]
    for c in range(cols):
      ax[c].set_title(subtitles[c])
      ax[c].set(xlabel='', ylabel=y_labels[c])
      plot(0, c, _ax)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.subplots_adjust(wspace=0.2)
    fig.show()

  def square_plots(self, plot, rows, cols, title, subtitles, x_labels, y_labels, figsize):
    fig, ax = plt.subplots(rows, cols, sharex=True, figsize=figsize)
    _ax = lambda x, y: ax[x, y]
    for r in range(rows):
      for c in range(cols):
        ax[r, c].set_title(subtitles[(r * cols) + c])
        ax[r, c].set(xlabel=x_labels[(r * cols) + c], ylabel=y_labels[(r * cols) + c])
        plot(r, c, _ax)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.show()

  def subplots(self, plot, rows, cols, title, subtitles, x_labels, y_labels, figsize):
    if(subtitles == ""):
      subtitles = [""] * rows * cols
    if(x_labels == ""):
      x_labels = [""]*rows*cols
    if(y_labels == ""):
      y_labels = [""]*rows*cols

    if(rows == 1 and cols==1):
      return self.single_plot(plot, title, x_labels, y_labels, figsize)
    if(rows == 1):
      return self.horizontal_plot(plot, cols, title, subtitles, x_labels, y_labels, figsize)
    elif(cols == 1):
      return self.vertical_plot(plot, rows, title, subtitles, x_labels, y_labels, figsize)
    else:
      return self.square_plots(plot, rows, cols, title, subtitles, x_labels, y_labels, figsize)

  def boxplot(self, df, y_value, title="", subtitles="", rows=1, cols=1, x_labels="", y_labels="", figsize=(8, 6)):
    plot = lambda r, c, ax: sns.boxplot(y=df[y_value[(r * cols) + c]], ax=ax(r, c))
    self.subplots(plot, rows, cols, title, subtitles, x_labels, y_labels, figsize)
  