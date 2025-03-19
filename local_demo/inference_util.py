import heapq

class MaxHeapDict:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # mapping of keys to entries
        self.REMOVED = '<removed>'  # placeholder for a removed item
        self.counter = 0  # unique sequence count

    def add_or_update(self, key, value):
        """ Add a new item or update the value of an existing item """
        if key in self.entry_finder:
            self.remove(key)
        entry = (-value, self.counter, key)
        self.entry_finder[key] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def remove(self, key):
        """ Mark an existing item as REMOVED. Raise KeyError if not found. """
        entry = self.entry_finder.pop(key)
        self.entry_finder[key] = self.REMOVED


    def pop_max(self):
        """ Remove and return the item with the highest priority """
        while self.heap:
            value, count, key = heapq.heappop(self.heap)
            if key is not self.REMOVED:
                del self.entry_finder[key]
                return key, -value
        raise KeyError('pop from an empty priority queue')

    def peek_max(self):
        """ Return the item with the highest priority without removing it """
        while self.heap:
            value, count, key = self.heap[0]
            if key is not self.REMOVED:
                return key, -value
            heapq.heappop(self.heap)  # remove stale entry
        raise KeyError('peek from an empty priority queue')

if __name__ == "__main__":

    # Example usage
    heap_dict = MaxHeapDict()

    data = {
        'apple': 10,
        'banana': 20,
        'cherry': 15,
        'date': 5
    }

    # Initial population of the heap
    for key, value in data.items():
        heap_dict.add_or_update(key, value)

    # Simulate a dictionary update
    data['banana'] = 25  # Update existing
    heap_dict.add_or_update('banana', 25)

    data['elderberry'] = 30  # Add new
    heap_dict.add_or_update('elderberry', 30)

    data.pop('apple')  # Remove existing
    heap_dict.remove('apple')

    # Pop elements from the heap
    while True:
        try:
            print(heap_dict.pop_max())
        except KeyError:
            break
