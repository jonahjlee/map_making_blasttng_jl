# --- NOTE: MOSTLY AI GENERATED CODE ---
# no guarantees this will work properly!

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class AnimatedScatterPlot(tk.Frame):
    def __init__(self, master, positions, timestreams, tick_ms, speed_mult=1, **kwargs):
        super().__init__(master, **kwargs)

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        assert set(positions.keys()) == set(timestreams.keys()), "positions and timestreams must have the same keys"

        self.speed_mult = speed_mult

        self.master = master
        self.positions = positions  # Dictionary {id: (x, y)}
        self.timestreams = timestreams  # Dictionary {id: [t1, t2, t3, ...]}
        self.tick_ms = tick_ms
        self.num_points = len(positions)
        self.tod_len = len(next(iter(self.timestreams.values())))  # All timestreams have equal length
        self.pack()

        self.dfs = lambda frame: [timestream[frame] for timestream in self.timestreams.values()]
        self.dfs_ctremoved = lambda frame: [timestream[frame] - np.mean(self.dfs(frame)) for timestream in self.timestreams.values()]

        self.x_data = [point[0] for point in self.positions.values()]
        self.y_data = [point[1] for point in self.positions.values()]

        self.scatter, self.colorbar, self.canvas, self.ani = None, None, None, None
        self.create_animated_plot(master=self)


    def create_animated_plot(self, master):
        # Create a Matplotlib figure and axis

        # Initialize scatter plot
        self.scatter = self.ax.scatter(self.x_data, self.y_data, c=np.zeros(self.num_points),
                                       s=200, cmap="viridis", marker='o', vmin=-0.5, vmax=0.5)
        self.ax.set_aspect('equal', adjustable='box')

        # Add color bar for reference
        self.colorbar = self.fig.colorbar(self.scatter)

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Set up the animation function
        self.ani = FuncAnimation(self.fig, self.update_plot, frames=self.tod_len,
                                 interval=self.tick_ms, repeat=False)

    def update_plot(self, frame):

        frame *= self.speed_mult

        # Update color array for scatter plot
        print(f'Index: {frame} / {self.tod_len} ({int(100 * frame / self.tod_len)}%)')
        self.scatter.set_array(np.array(self.dfs_ctremoved(frame)))  # Update color data
        self.canvas.draw()  # Redraw the canvas

        return [self.scatter]  # list of artists that were updated


if __name__ == '__main__':

    # Example data for demonstration
    pos_dict = {
        "A": (1, 2),
        "B": (2, 3),
        "C": (3, 1),
        "D": (4, 4),
        "E": (5, 2)
    }

    # Random example timestreams for each point
    tod_dict = {
        "A": np.random.rand(100),  # Timestreams with 10 time points
        "B": np.random.rand(100),
        "C": np.random.rand(100),
        "D": np.random.rand(100),
        "E": np.random.rand(100)
    }

    # Main Tkinter window setup
    root = tk.Tk()
    root.title("Animated Scatter Plot")

    app = AnimatedScatterPlot(master=root, positions=pos_dict, timestreams=tod_dict, tick_ms=100)

    # Start the GUI loop
    app.mainloop()
