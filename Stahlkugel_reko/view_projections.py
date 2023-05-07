#!/home/gcg/miniconda3/envs/tigre/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys


class IndexTracker(object):
    def __init__(self, X):

        self.fig, self.ax = plt.subplots(1, 1)
        # adjust the main plot to make room for the sliders
        self.fig.subplots_adjust(bottom=0.25)

        self.ax.set_title('projections')

        self.X = X
        self.slices, cols, rows,  = X.shape
        self.ind = self.slices//2

        self.im = self.ax.imshow(self.X[self.ind, :, :], origin="lower")


        # Make a horizontally oriented slider to control the amplitude
        axamp = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.slice_slider = Slider(
            ax=axamp,
            label="slice",
            valmin=0,
            valmax=self.slices,
            valinit=self.ind,
        )
        # register the update function with each slider
        self.slice_slider.on_changed(self.update)
        self.update()
        self.fig.colorbar(self.im, ax=self.ax, anchor=(0, 0.3))
        plt.show()


    def update(self, tmp=1.0):
        self.ind = min(int(self.slice_slider.val), self.slices - 1) # limit to valid range
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

if __name__ == "__main__":
    print("arguments", sys.argv)
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("need to run this tool like:")
        print("view_projections projections.npy")
        sys.exit()

    with open(filename, 'rb') as f:
        P = np.load(f)
    slicer = IndexTracker(P)


