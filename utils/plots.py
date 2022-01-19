import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def figure(width=None, height=None, xrange=None, yrange=None, **kwargs):
    layout = {
        'xaxis': dict(range=xrange),
        'yaxis': dict(range=yrange),
        'template': 'plotly_dark',
        'height': height,
        'width': width,
        'margin': dict(l=10, r=10, t=10, b=10),
        **kwargs
    }
    fig = go.FigureWidget(layout=layout)
    return fig


def subplots(rows=2, cols=2, width=None, height=None, xrange=None, yrange=None, titles=None):
    fig = make_subplots(rows=rows, cols=cols,
                        vertical_spacing=0.2, horizontal_spacing=0.05,
                        subplot_titles=titles)
    fig.update_layout(
        template='plotly_dark', showlegend=False, height=height, width=width,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)
    return fig


def dark_layout(xrange=None, yrange=None, **kwargs):
    layout = dict(
        template='plotly_dark',
        **kwargs
    )
    if xrange:
        layout['xaxis_range'] = xrange
    if yrange:
        layout['yaxis_range'] = yrange
    return layout

def dark_layout_3d(xrange=None, yrange=None, zrange=None):
    layout = dict(
        showlegend=True,
        template='plotly_dark',
        margin=dict(l=20, r=50, t=20, b=20),
        scene=dict()
    )
    if xrange:
        layout['scene']['xaxis_range'] = xrange
    if yrange:
        layout['scene']['yaxis_range'] = yrange
    if zrange:
        layout['scene']['zaxis_range'] = zrange

    return layout

def grid_points(start, stop, step):
    ax1, ax2 = np.mgrid[start:stop:step, start:stop:step]
    points = np.dstack([ax1, ax2]).reshape(-1, 2)
    return ax1.flatten(), ax2.flatten(), points