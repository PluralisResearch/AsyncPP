# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
from collections import deque

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""
class Queue:
    def __init__(self):
        self.queue = deque()
        self.cv = threading.Condition()

    def add(self, tensor):
        self.cv.acquire()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.popleft()
        self.cv.release()
        return tensor
