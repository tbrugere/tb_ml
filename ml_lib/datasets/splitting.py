import numpy as np

def split_indices(num_samples, *percents):
    if not isinstance(num_samples, int):
        num_samples = len(num_samples)
    assert np.isclose(sum(percents) , 1)
    cdf = np.cumsum(percents)
    cdf = num_samples * cdf
    shuffle = np.arange(num_samples)
    rng.shuffle(shuffle)
    indices = (shuffle[:, None] >= cdf[None, :])
    indices = indices.sum(-1)
    return indices
    
def split_arrays(split_indices, *arrays):
    all_splits = np.unique(split_indices)
    
    ret = []
    for i in all_splits:
        mask = split_indices == i
        if len(arrays) == 1:
            array, = arrays
            ret.append(array[mask])
        else:
            ret.append([array[mask] for array in arrays])
    if len(ret) == 1:
        ret, = ret
    return ret
