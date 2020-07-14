import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame="polygon"):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = "radar"
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


class _Radarplotter:
    def __init__(
        self,
        data=None,
        facets=None,
        normalize=True,
        color=None,
        alpha=None,
        figsize=None,
        ax=None,
        label=None,
        title=None,
        frame=None,
        grid=None,
    ):
        assert isinstance(normalize, bool)

        self.plot_data = self._get_data(data, facets, normalize)
        self.facets = facets
        self.n = len(facets)
        self.color = color
        self.alpha = alpha
        self.title = title
        self.frame = frame
        self.grid = grid
        self.theta = radar_factory(self.n, self.frame)
        self.ax = (
            ax
            if ax
            else plt.subplots(figsize=figsize, subplot_kw=dict(projection="radar"))[1]
        )
        angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.angles = angles + angles[:1]

    def _get_data(self, data, facets, normalize):
        if normalize:
            return data[facets] / data[facets].max(axis=1)[0]
        return data[facets]

    def _plot(self):
        theta = radar_factory(self.n, frame=self.frame)
        self.ax.plot(theta, self.plot_data.values[0], color=self.color)
        self.ax.fill(
            theta, self.plot_data.values[0], facecolor=self.color, alpha=self.alpha
        )

        self.ax.set_title(
            self.title,
            weight="bold",
            size="large",
            position=(0.5, 1.1),
            horizontalalignment="center",
            verticalalignment="center",
        )
        self.ax.set_varlabels(self.facets)

        if self.grid:
            self.ax.set_rgrids(np.linspace(0, self.plot_data.max().max(), self.n))

        return self.ax


def plot(
    data=None,
    facets=None,
    normalize=True,
    color="red",
    alpha=0.5,
    figsize=None,
    ax=None,
    title=None,
    frame="polygon",
    grid=True,
):

    plotter = _Radarplotter(
        data=data,
        facets=facets,
        normalize=normalize,
        color=color,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        frame=frame,
        title=title,
        grid=grid,
    )

    return plotter._plot()
