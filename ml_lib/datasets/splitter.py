import functools as ft

class SplitDataset():

    train: Optional[torch.DataLoader]
    validate: Optional[torch.DataLoader]
    test: Optional[torch.Dataloader]



    def __init__(self, train, validate, test):
        pass

