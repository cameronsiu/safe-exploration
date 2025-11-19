import time

class FrequencyCounter:
    def __init__(self):
        self._length = 50
        self._timings = [None] * self._length
        self._index = 0

    def record(self):
        self._timings[self._index] = time.time()
        self._index = self._get_index(self._index + 1)

    def frequency(self):
        timings = [t for t in self._timings if t is not None]
        num_events = len(timings)
        start = timings[self._get_index(self._index, num_events)]
        end = timings[self._get_index(self._index - 1, num_events)]
        duration = end - start
        if duration == 0:
            return 0
        else:
            return num_events / duration

    def _get_index(self, index, length = None):
        length = length or self._length
        if index == -1:
            index = length - 1
        return index % length
