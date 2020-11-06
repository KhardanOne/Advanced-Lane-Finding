import numpy as np
from globals import *

class Line:
    def __init__(self, frame_count, side, image_height, fit, quality, all_raw_x=None, all_raw_y=None):
        # quality of the line - depending on the way the fit was aquired
        # 0 - guessed by old history
        # 1 - quessed by recent history
        # 2 - guessed by the other lane, which is of quality 1 (but not 0, to be evaluated)
        # 3 - guessed by the other lane, which is of quality 4 or above
        # 4 - so-so fit found
        # 5 - perfect fit
        self.quality = quality
        self.side = side                    # 'left' or 'right'
        self.frame = frame_count            # in which frame was the line last time updated
        self.coeffs = fit                   # polynomial coefficients
        self.all_raw_x = all_raw_x          # x values for raw pixels
        self.all_raw_y = all_raw_y
        self.image_height = image_height    # used for sanity checks

    def get_x_for_y(self, y):
        x = self.coeffs[0] * y ** 2 + self.coeffs[1] * y + self.coeffs[2]
        return x

    def is_sane_parallel(self, other_lane):
        """Checks the lane width does not vary too much with y."""
        ys = np.array([0.1, 0.5, 0.9]) * self.image_height
        xs = np.array([self.get_x_for_y(ys[0]), self.get_x_for_y(ys[1]), self.get_x_for_y(ys[2])])
        xos = np.array([other_lane.get_x_for_y(ys[0]), other_lane.get_x_for_y(ys[1]), other_lane.get_x_for_y(ys[2])])
        if self.side == 'left':
            dists = xos - xs
        else:
            dists = xs - xos
        deviations = dists.max() - dists.min()
        sane = 1 if deviations < 250 else 0  # in pixels in 2D  ############################################# tune here
        return sane, deviations

    def is_sane_other_dist(self, other_lane):
        """Checks the lane width is between same limits at three different y."""
        ys = np.array([0.1, 0.5, 0.9]) * self.image_height
        xs = np.array([self.get_x_for_y(ys[0]), self.get_x_for_y(ys[1]), self.get_x_for_y(ys[2])])
        xos = np.array([other_lane.get_x_for_y(ys[0]), other_lane.get_x_for_y(ys[1]), other_lane.get_x_for_y(ys[2])])
        if self.side == 'left':
            dists = xos - xs
        else:
            dists = xs - xos
        min = dists.min()
        max = dists.max()
        sane = 1 if min > 700 and max < 1200 else 0  # in pixels in 2D  ##################################### tune here
        return sane, min, max
