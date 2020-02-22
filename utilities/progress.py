import math
from time import time


class Progress:
    """Progress bar with length :length: and :total: number of iterations."""

    def __init__(self, total, length=80, name="Progress:"):
        self.total = total
        self.length = length
        self.start = time()
        self.ipc = total / length
        self.i = 0
        self.name = name

    def update(self, i):
        filled_count = math.floor(self.length * i / self.total)
        filled_portion = "#" * filled_count
        remaining_portion = "." * (self.length - filled_count)
        time_elapsed = time() - self.start
        time_estimate = time_elapsed * (self.total / i) if i > 0 else math.inf
        print(f"\r{self.name} [{filled_portion}{remaining_portion}] Time: {time_elapsed:.1f} / {time_estimate:.1f} s",
              end='', flush=True)
        self.i = i

    def increment(self):
        self.update(self.i + 1)

    def done(self):
        self.update(self.total)
        print()
