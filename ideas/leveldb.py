

class Memtable:

    def __init__(self, max_length: int = 10):
        self.mutable = True
        self.store = {}
        self.length = 0
        self.max_length = max_length

    def put(self, key, value):
        self.store[key] = value
        self.length += 1
        if self.length > self.max_length:
            self.flush()

    def get_by_key(self, key):
        return self.store.get(key, None)
    
    def delete_by_key(self, key):
        if key in self.store:
            self.store[key] = "DELETION_MARK"
        else:
            self.put(key, "DELETION_MARK")
            self.length += 1
        if self.length > self.max_length:
            self.flush()

    def flush(self):
        self.mutable = False


class SSTable:

    def __init__(self, max_length: int):
        self.max_length = max_length
        self.store = {}
        self.length = 0

    def put(self, key, value):
        pass

    def get_by_key(self, key):
        pass

    def delete_by_key(self, key):
        pass


class FakeLevelDB:

    def __init__(self, num_layers: int, base_size: int = 5):
        self.num_layers = num_layers
        self.base_size = base_size
        self.layers = {}
        for i in range(num_layers):
            self.layers.append(([], base_size * (i + 1)))
    
    def put(self, key, value):
        cur_size = len(self.layers[0])
        for memtable in self.layers[0]:
            if memtable.mutable:
                memtable.put(key, value)
                return
        # if no mutable memtable and layer 0 is not full, then create one
        if cur_size < self.layers[0][1]:
            memtable = Memtable()
            memtable.put(key, value)
            self.layers[0][0].append(memtable)
            return
        # if layer 0 is full, then move to layer 1
        if cur_size >= self.layers[0][1]:
            self.layers[1][0].append(Memtable())
            self.layers[1][0][-1].put(key, value)
            return

    def get_by_key(self, key):
        pass

    def delete_by_key(self, key):
        pass

    def compaction(self):
        pass

