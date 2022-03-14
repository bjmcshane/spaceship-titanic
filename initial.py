import pandas as pd
import numpy as np


training = pd.read_csv("data/train.csv")
testing = pd.read_csv("data/test.csv")



print(training.shape)
print(training.index)
print(training.columns)
training.info()
training.describe()
