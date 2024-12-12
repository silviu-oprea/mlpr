import matplotlib.pyplot as plt


def reset_plot(xrange=(-3, 3), size=(5, 4), yrange=None):
    plt.cla()
    if yrange is None:
        yrange = xrange
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    plt.rcParams['figure.figsize'] = size
    plt.rcParams['text.usetex'] = True
