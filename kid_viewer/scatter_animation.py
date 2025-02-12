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
                 master: tk.Tk,
                 positions: dict[int, tuple[float, float]],
                 timestreams: dict[int, np.ndarray],
                 tick_ms: int,
                 speed_mult: int=1,
                 **kwargs):
        super().__init__(master, **kwargs)

        assert set(positions.keys()) == set(timestreams.keys()),\
            "positions and timestreams must have the same keys"

        self.master: tk.Tk = master
        self.positions: dict[int, tuple[float, float]] = positions
        self.timestreams: dict[int, np.ndarray] = timestreams
        self.tick_ms: int = tick_ms
        self.speed_mult: int = speed_mult

        self.num_points: int = len(positions)
        self.tod_len: int = len(next(iter(self.timestreams.values())))  # All timestreams have equal length

        self.pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.x_data: list = [point[0] for point in self.positions.values()]
        self.y_data: list = [point[1] for point in self.positions.values()]
        self.color_vals = lambda i: [timestream[i] for timestream in self.timestreams.values()]

        self.scatter, self.colorbar, self.canvas, self.ani = None, None, None, None
        self.create_animated_plot()


    def create_animated_plot(self):
        self.scatter = self.ax.scatter(
            self.x_data,
            self.y_data,
            s=200, marker='h',
            c=np.zeros(self.num_points), cmap="coolwarm",
            norm=colors.SymLogNorm(0.001)
        )
        self.ax.set_aspect('equal', adjustable='box')
        self.colorbar = self.fig.colorbar(self.scatter)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=self.tod_len,
                                 interval=self.tick_ms, repeat=False)

    def update_plot(self, frame):

        frame *= self.speed_mult

        print(f'Index: {frame} / {self.tod_len} ({int(100 * frame / self.tod_len)}%)')

        self.scatter.set_array(
            np.array(self.color_vals(frame))
        )
        self.canvas.draw()

        return [self.scatter]  # list of artists that were updated


# ============================================================================ #
# EXAMPLE USAGE
# ============================================================================ #

if __name__ == '__main__':

    # Example data for demonstration
    pos_dict = {
        1: (1., 2.),
        2: (2., 3.),
        3: (3., 1.),
        4: (4., 4.),
        5: (5., 2.)
    }

    # Random example timestreams for each point
    tod_dict = {
        1: np.random.rand(100),  # Timestreams with 10 time points
        2: np.random.rand(100),
        3: np.random.rand(100),
        4: np.random.rand(100),
        5: np.random.rand(100)
    }

    # Main Tkinter window setup
    root = tk.Tk()
    root.title("Animated Scatter Plot")

    app = AnimatedScatterPlot(master=root, positions=pos_dict, timestreams=tod_dict, tick_ms=100)

    # Start the GUI loop
    app.mainloop()
