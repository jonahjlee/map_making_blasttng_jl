# ============================================================================ #
# scatter_animation.py
#
# Jonah Lee (with use of generative AI)
#
# Scatter Animation
# Tkinter widget: display a scatter plot as an animation over time.
# Allows assignment of color for each marker for each frame (timestamp)
# ============================================================================ #


# ============================================================================ #
# IMPORTS
# ============================================================================ #

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================================ #
# AnimatedScatterPlot
# ============================================================================ #

class AnimatedScatterPlot(tk.Frame):
    def __init__(self,
                 parent,
                 positions: dict[str, tuple[float, float]],
                 timestreams: dict[str, np.ndarray],
                 tick_ms: int,
                 speed_mult: int=1,
                 slider: tk.Scale=None,
                 **kwargs):
        super().__init__(parent, **kwargs)

        assert set(positions.keys()) == set(timestreams.keys()), \
            "positions and timestreams must have the same keys"

        self.parent: tk.Tk = parent
        self.positions: dict[str, tuple[float, float]] = positions
        self.timestreams: dict[str, np.ndarray] = timestreams
        self.tick_ms: int = tick_ms
        self.speed_mult: int = speed_mult

        self.slider = None
        self.slider_val = None
        self.set_slider(slider)

        self.num_points: int = len(positions)
        self.tod_len: int = len(next(iter(self.timestreams.values())))  # all timestreams have equal length

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.tight_layout()
        self.ax.set_aspect('equal')

        self.x_data: list = [point[0] for point in self.positions.values()]
        self.y_data: list = [point[1] for point in self.positions.values()]
        self.color_vals = lambda i: [timestream[i] for timestream in self.timestreams.values()]

        self.scatter, self.colorbar, self.canvas, self.ani = None, None, None, None

        self._is_playing = True
        self.frame = 0
        self.create_animated_plot()

    def create_animated_plot(self):
        self.scatter = self.ax.scatter(
            self.x_data,
            self.y_data,
            s=350, marker='h',
            c=np.zeros(self.num_points), cmap="viridis",
            vmin=-0.3, vmax=0.5
        )
        self.colorbar = self.fig.colorbar(self.scatter)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=self.get_frame,
                                 interval=self.tick_ms, repeat=True, cache_frame_data=False)

    def update_plot(self, frame):

        fraction_done = frame * self.speed_mult / self.tod_len
        if self.slider is not None:
            self.slider_val.set(fraction_done)

        # print(f'Index: {frame} / {self.tod_len} ({int(100 * fraction_done)}%)')

        self.scatter.set_array(
            np.array(self.color_vals(frame))
        )
        self.canvas.draw()

        return [self.scatter]  # list of artists that were updated

    @property
    def is_playing(self):
        return self._is_playing

    def pause(self):
        self.ani.pause()
        self._is_playing = False

    def resume(self):
        self.ani.resume()
        self._is_playing = True

    def get_frame(self):
        while self.frame < self.tod_len:
            yield self.frame
            self.frame += self.speed_mult

    def set_slider(self, slider: tk.Scale):
        if slider is not None:
            self.slider = slider
            self.slider_val = tk.DoubleVar()  # must use variable or else set() calls self.slider_update
            self.slider.config(command=self.slider_update, variable=self.slider_val)

    def slider_update(self, value: str):
        self.frame = int(float(value) * self.tod_len)
        if not self.is_playing:
            # update the image
            # could unpause-resume to avoid calling protected method?
            self.ani._draw_frame(self.frame)
