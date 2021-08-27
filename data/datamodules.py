from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import csv

class Moons4D2D(LightningDataModule):
    def __init__(self):
        super().__init__()
        Xtrain = np.zeros((500, 4))
        with open('data/data_v2.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for i, row in enumerate(reader):
                if i!=0:
                    Xtrain[i-1] = [float(r) for r in row[1:]]
        Xtest = np.zeros((500, 4))
        with open('data/data_v2_eval.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for i, row in enumerate(reader):
                if i!=0:
                    Xtest[i-1] = [float(r) for r in row[1:]]
        Xtrain = Xtrain.astype('float32')
        test = Xtest.astype('float32')
        
        train, val = train_test_split(Xtrain, test_size=0.05)
        self.train_data = TensorDataset(train)
        self.val_data = TensorDataset(val)
        self.test_data = TensorDataset(test)
        
    def train_dataloader(self):
        return DataLoader(self.train_data)
        
    def val_dataloader(self):
        return DataLoader(self.val_data)
        
    def test_dataloader(self):
        return DataLoader(self.test_data)
        
        
if __name__ == '__main__':
    datamodule = Moons4D2D()
    