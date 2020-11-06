import numpy as np
from globals import *

class LineHistory:
    def __init__(self, side):
        self.side = side
        self.reset()

    def reset(self):
        self.history = [None] * (CFG['history_length'] + 1)  # +1 to allow for easier iterating over the next items

    def get_future_coeffs(self, for_frame):
        """
        !!! Not implemented !!!
        Calculates fit from old fit and old derived fit.
        Fails if the history doesn't contain enough data for the time periods.
        """
        raise NotImplemented
        pass

    def _get_avg_coeffs(self, quality_limit=4):
        """
        Don't forget to remove the last element after using this!
        Returns True if avg is found, False otherwise. Second return value is the coeffs np.array.
        """
        sums = np.array([0] * 3, dtype=np.float64)
        count = 0
        for line in self.history:
            if line is not None and line.quality >= quality_limit:
                sums += line.coeffs
                count += 1
        if count > 0:
            avg = sums / count
            return True, avg
        else:
            return False, np.array([0., 0., 80])

    def get_avg_coeffs_and_remove(self, quality_limit=4):
        avg = self._get_avg_coeffs(quality_limit)
        self._delete_oldest()
        return avg

    def get_raw_xy(self):
        """!!! Not implemented !!! Return last x y raw data. Improved version: use derivatives for prediction."""
        raise NotImplemented
        pass

    def is_sane_avg(self, future_fit):
        """False if it deviates from the stored avg. Returns True if there is no enough data or it is OK."""
        if self._get_size() < 3: ####################################################################################### tune here
            return True, np.array([0., 0., 0.]), np.array([0., 0., 0.])
        limits = np.array([0.0003, 0.5, 180.])  ######################################################################## tune here
        has_avg, avg = self._get_avg_coeffs()
        if not has_avg:  # not enough data yet
            return True, np.array([0., 0., 0.]), np.array([0., 0., 0.])
        diffs = np.abs(future_fit - avg)
        result = 1 if np.less_equal(diffs, limits).all() else 0
        return result, avg, diffs

    def is_sane_future(self, future_fit):
        """!!! Not implemented !!! Like is_sane_avg but it also takes into account the derived fit."""
        raise NotImplemented
        pass

    def update(self, line, lowest_quality_for_avg=4):
        """Stores the line and returns the stored average."""
        line.side = line.side if line.side else self.side
        self._shift_stored_data()
        self.history[0] = line
        avg = self._get_avg_coeffs(lowest_quality_for_avg)
        if not avg:  # if there is not enough good quality stored data but we have a current fit, then use the current, this should never happen
            avg = line.coeffs
        return avg

    def _get_size(self):
        count = 0
        for i in range(CFG['history_length']):  # optimization possibility: store the size
            if self.history[i]:
                count += 1
        return count

    def _shift_stored_data(self):
        """Stores all elements making a free space (that is, None) at idx 0."""
        for dst in reversed(range(CFG['history_length'])):  # optimization possibility: circular indices
            src = dst - 1
            self.history[dst] = self.history[src]
        self.history[0] = None

    def _delete_oldest(self):
        for target in reversed(range(CFG['history_length'])):  # optimization possibility: store the index
            if self.history[target]:
                self.history[target] = None
                break
