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