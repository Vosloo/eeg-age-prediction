from src.data_loader import DataLoader
import pandas as pd


if __name__ == '__main__':
    for measurement in DataLoader.get_eval():
        print(measurement)
        print(measurement.data.values.shape)
        print(measurement.data.values.flatten().T.shape)
        break    

