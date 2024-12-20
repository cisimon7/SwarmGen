import numpy as np
from math import floor

import plotly
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plotly.offline.plot
pio.renderers.default = "vscode"
cols = plotly.colors.qualitative.Plotly


def multiviz_trajs(traj_x, traj_y, traj_z, nrows=2, ncols=2, num_agents=16, height=400, width=1_600, titles=[]):
    fig = make_subplots(
        rows=nrows, cols=ncols, horizontal_spacing=0.05, vertical_spacing=0.05,
        subplot_titles=titles,
        specs=[[{'type': 'scatter3d'} for _ in range(ncols)] for _ in range(nrows)]
    )

    for i in range(int(nrows*ncols)):
        for j in range(num_agents):
            fig.add_trace(
                go.Scatter3d(x=traj_x[i, j], y=traj_y[i, j], z=traj_z[i, j], mode='lines', showlegend=False),
                row=1+floor(i/ncols), col=1+(i%ncols)
            )

        fig.add_trace(
            go.Scatter3d(x=traj_x[i, :, 0], y=traj_y[i, :, 0], z=traj_z[i, :, 0], mode='markers', showlegend=False, marker=dict(size=2, color=cols[0])),
            row=1+floor(i/ncols), col=1+(i%ncols)
        )
        fig.add_trace(
            go.Scatter3d(x=traj_x[i, :, -1], y=traj_y[i, :, -1], z=traj_z[i, :, -1], mode='markers', showlegend=False, marker=dict(size=2, color=cols[1])),
            row=1+floor(i/ncols), col=1+(i%ncols)
        )

    fig.update_layout(height=height, width=width, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=1),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        zaxis=dict(showticklabels=False, title=""),
    )
    fig.update_xaxes(showline=True, mirror=True)
    fig.update_yaxes(showline=True, mirror=True)
    fig.show()



def single_view(traj_x, traj_y, traj_z, opacity=1, nrows=2, ncols=2, num_agents=16, height=1_200, width=1_200):
    cols = plotly.colors.qualitative.Pastel
    cols1 = plotly.colors.qualitative.Dark24

    fig = make_subplots(
        rows=nrows, cols=1, horizontal_spacing=0.05, vertical_spacing=0.05,
        specs=[[{'type': 'scatter3d'} for _ in range(1)] for _ in range(nrows)]
    )

    if type(opacity) == int or type(opacity) == float:
        opacity = np.ones(num_agents) * opacity

    for i in range(int(nrows*ncols)):
        for j in range(num_agents):
            fig.add_trace(
                go.Scatter3d(x=traj_x[i, j], y=traj_y[i, j], z=traj_z[i, j], mode='lines', opacity=opacity[j], showlegend=False, line=dict(width=7, color=cols[(i+2)%10])),
                row=1+floor(i/ncols), col=1#+(i%ncols)
            )
        fig.add_trace(
            go.Scatter3d(x=traj_x[i, :num_agents, 0], y=traj_y[i, :num_agents, 0], z=traj_z[i, :num_agents, 0], mode='markers', showlegend=False, marker=dict(symbol="square", size=5, color=cols1[-1])),
            row=1+floor(i/ncols), col=1#+(i%ncols)
        )
        fig.add_trace(
            go.Scatter3d(x=traj_x[i, :num_agents, -1], y=traj_y[i, :num_agents, -1], z=traj_z[i, :num_agents, -1], mode='markers', showlegend=False, marker=dict(symbol="x", size=3, color=cols1[-2])),
            row=1+floor(i/ncols), col=1#+(i%ncols)
        )

    fig.update_layout(
        template="seaborn", 
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=height, width=width, 
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            # up=dict(x=0, y=0, z=1),
            # center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=-1.25, z=1.25)
        ),
        xaxis=dict(showticklabels=False, title="", tickfont=dict(family='Helvetica', size=14, weight="bold")),# range=[-1.5, 2]),
        yaxis=dict(showticklabels=False, title="", tickfont=dict(family='Helvetica', size=14, weight="bold")),# range=[-0.5, 2]),
        zaxis=dict(showticklabels=False, title="", tickfont=dict(family='Helvetica', size=14, weight="bold")),# range=[-2, 2]),
    )
    fig.update_xaxes(showline=True, mirror=True)
    fig.update_yaxes(showline=True, mirror=True)
    fig.show()