import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import StandardScaler

class DatasetReader:
    def __init__(self, filename: str = "monojet_Zp2000.0_DM_50.0_chan3.csv"):
        self.filename = filename
        self.cols = ["event_ID", "process_ID", "event_weight", "MET", "MET_Phi"]
        self.ignore_particles = ["e-", "e+", "m-", "m+", "g", "b"]

    def readFile(self) -> List[List[str]]:
        data = []
        print("Opening file: " + self.filename)
        file = open(self.filename, "r")

        print("Reading file line by line...")
        for line in file.readlines():
            cleaned_line = line.replace(";", ",")
            cleaned_line = cleaned_line.rstrip(",\n")
            cleaned_line = cleaned_line.split(",")
            data.append(cleaned_line)
        print(type(data))
        return data

    def createDataFrame(self, data: List[List[str]]):
        longest_line = max(data, key=len)

        n_max_cols = len(longest_line)
        print("Number of maximum possible columns: " + str(n_max_cols))

        print("Our cols are: " + str(self.cols))
        print("Creating deep copy of cols")
        copy_cols = self.cols.copy()

        for i in range(1, (int((n_max_cols - 5) / 5)) + 1):
            self.cols.append("obj_" + str(i))
            self.cols.append("E_" + str(i))
            self.cols.append("pt_" + str(i))
            self.cols.append("eta_" + str(i))
            self.cols.append("phi_" + str(i))

        print("Number of cols: " + str(len(self.cols)))
        print("\nSlicing list of cols: " + str(self.cols[50:60]))

        df = pd.DataFrame(data, columns=self.cols)
        df.fillna(value=np.nan, inplace=True)

        df_data = pd.DataFrame(df.values, columns=self.cols)
        df_data.fillna(value=0, inplace=True)
        df_data = df_data.drop(columns=copy_cols, inplace=True)

        ignore_list = []
        for i in range(len(df_data)):
            for j in data.loc[i].keys():
                if "obj" in j:
                    if data.loc[i][j] in self.ignore_particles:
                        ignore_list.append(i)
                        break

        df_data = df_data.drop(ignore_list, inplace=True)

        x = df_data.values.reshape([df_data.shape[0] * df_data.shape[1] // 5, 5])

        temp_list = []
        for i in range(x.shape[0]):
            if (x[i] == 0).all():
                temp_list.append(i)        
        x1 = np.delete(x, temp_list, 0)
        del x

        temp_list = []
        for i in range(x1.shape[0]):   
            if  (x1[i][0] == 'j'):
                continue
            else:
                temp_list.append(i)
                print(i, x1[i][0])
        
        data = np.delete(x1, temp_list, 0)

        col_names = ['obj', 'E', 'pt', 'eta', 'phi']
        data_df = pd.DataFrame(data, columns=col_names)
        # Drop the 'obj' column as it's unnecessary
        data_df.drop(columns='obj', inplace=True)
        data_df = data_df.astype('float32')

        return data_df

if __name__ == "__main__":
    dr = DatasetReader()
    data = dr.readFile()
    data_df = dr.createDataFrame(data)
