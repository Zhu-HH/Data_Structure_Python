"""
Python 实现 Queue
"""
class Queue():

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self,item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop() if self.items else None

    def size(self):
        return len(self.items)