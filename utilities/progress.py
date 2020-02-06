import math
from time import time


class Progress:
    """Progress bar with :length: and :total: number of iterations."""

    def __init__(self, total, length=80, name="Progress:"):
        self.total = total
        self.length = length
        self.start = time()
        self.ipc = total / length
        self.i = 0
        self.name = name

    def update(self, i):
        print("\r{} [{}{}] Time: {:.1f} / {:.1f} s"
              "".format(self.name, "#" * int(i // self.ipc), "." * int(self.length - i // self.ipc - 1),
                        time() - self.start, (time() - self.start) * (self.total / i) if i > 0 else math.inf),
              end='', flush=True)
        self.i = i

    def increment(self):
        self.update(self.i + 1)

    def done(self):
        print("\r{} [{}] Time: {:.1f} / {:.1f} s\n"
              "".format(self.name, "#" * (self.length - 1), time() - self.start, time() - self.start),
              end='', flush=True)
