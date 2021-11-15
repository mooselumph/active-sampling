import numpy as np



class Dataset():
    def __init__(self, x, y): 
        self.x = x
        self.y = y

    def __len__(self): 
        return len(self.x)

    def __getitem__(self, i): 
        return self.x[i],self.y[i]


class Sampler():
    def __init__(self, ds, bs, shuffle=False, seed=None):
        self.n = len(ds)
        self.bs = bs
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=seed)
        
    def __iter__(self):
        self.idxs = self.rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        for i in range(0, self.n, self.bs):
            yield self.idxs[i:i+self.bs]


class DataLoader():
    def __init__(self, ds, sampler):
        self.ds = ds
        self.sampler = sampler
        
    def __iter__(self):
        for s in self.sampler:
            yield self.collate([self.ds[i] for i in s])

    def collate(self, b):
        xs,ys = zip(*b)
        return np.stack(xs),np.stack(ys)





def get_train_test_dataloader(train_ds, test_ds, config):

    """
    Params:
    train_ds: tuple of np.arrays, where first element is all the training data, second element is corresponding labels
    test_ds: same thing but for test data
    config: hydra config object

    Returns:
    trainloader, testloader: Dataloaders as defined above
    """

    x_train, y_train = train_ds
    x_test, y_test = test_ds

    train_ds = Dataset(x_train, y_train)
    test_ds = Dataset(x_test, y_test)

    shuffle_train = not config.run.deterministic
    seed = None if shuffle_train else 42
    train_sampler = Sampler(train_ds, config.train.batch_size, shuffle=shuffle_train, seed=seed)
    trainloader = DataLoader(train_ds, train_sampler)

    test_sampler = Sampler(test_ds, config.train.batch_size, shuffle=False)
    testloader = DataLoader(test_ds, test_sampler)

    return trainloader, testloader

