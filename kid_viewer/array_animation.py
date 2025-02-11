import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class ArrayAnimation(tk.Frame):
    def __init__(self, master, shifts, tods):
        super().__init__(master)
        assert set(shifts.keys()) == set(tods.keys()), "shifts and tods must have the same keys!"
        self.master = master  # tkinter parent widget
        self.shifts: dict[int, tuple[float, float]] = shifts
        self.tods: dict[int, np.ndarray] = tods
        self.kid_ids: list[int] = list(self.shifts.keys())
        self.detector_count: int = len(self.kid_ids)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Create a Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(5, 4))

        # Initial plot
        self.scatter_plot = self.ax.scatter()

        # Embed Matplotlib figure in Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Set up the animation function
        self.ani = FuncAnimation(self.fig, self.update_plot, frames=np.linspace(0, 2*np.pi, 128),
                                 interval=1)

    def update_plot(self, frame):
        self.y_data = np.sin(self.x_data + frame)
        self.scatter_plot.set_ydata(self.y_data)  # Update the data of the line
        self.canvas.draw()  # Redraw the canvas

