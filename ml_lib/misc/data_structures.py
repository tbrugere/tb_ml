

############### dict functions

def merge_dicts(*dicts: dict) -> dict:
    """merge dictionaries

    merges the dicts, giving precedence to the last one, RECURSIVELY

    I use this to merge configurations 
    (so say you’ve loaded a default config with all fields, 
     and a user config with some modified fields, you can merge them)

    Args:
        dicts: dictionaries

    Returns:
        dict: a dictionary containing all the keys in the dictionaries, with the value of the last argument containing that key
    """
    merged = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict) and key in merged:
                value = merge_dicts(merged[key], value)
            merged[key] = value

    return merged


def unwrap_dict(d: dict, *keys):
    """[TODO:summary]

    asserts that the keys in d are the ones in the keys list, 
    and returns the elements in that order

    Args:
        d: [TODO:description]
    """
    assert set(d.keys()) == set(keys), f"wrong set of keys, got {d.keys()}, expected {keys}"
    return [d[key] for key in keys]



############### Union-Find

class UnionFind(list):
    """Union-find data structure with path compression"""
    def __init__(self, n):
        super().__init__(range(n))

    def _get_parent(self, k):
        return super().__getitem__(k)

    def __getitem__(self, k):
        pk = self._get_parent(k)
        if pk == k:
            return pk
        cl = self[pk]
        super().__setitem__(k, cl)
        return cl

    def __setitem__(self, k, l):
        super().__setitem__(k, self[l])

    def union(self, k, l):
        """merges the class of k into that of l"""
        self[self[k]] = self[l]

class UnionFindNoCompression(list):
    """Union-find data structure without path compression"""
    def __init__(self, n):
        super().__init__(range(n))

    def _get_parent(self, k):
        return super().__getitem__(k)

    def __getitem__(self, k):
        pk = self._get_parent(k)
        if pk == k:
            return pk
        cl = self[pk]
        #super().__setitem__(k, cl)#path compression
        return cl

    def __setitem__(self, k, l):
        super().__setitem__(k, l)

    def union(self, k, l):
        """merges the class of k into that of l"""
        self[self[k]] = l

    def reroot(self, k, _previous=None):
        """changes the representant of k’s class to k"""
        if _previous is None:
            _previous = k
        p = self._get_parent(k)
        super().__setitem__(k, _previous)
        if p == k:
            return
        self.reroot(p, _previous=k)
