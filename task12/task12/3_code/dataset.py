import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, x_dir, y_dir): # x_dir is train data path, y_dir is train label path
        self.x = np.array(pd.read_csv(x_dir, header=None), dtype=np.float32)
        self.y = np.array(pd.read_csv(y_dir, header=None), dtype=np.int).squeeze(-1)

        ###print()
        # print('np.array(pd.read_csv(x_dir, header=None), dtype=np.float32)', np.array(pd.read_csv(x_dir, header=None), dtype=np.float32))
        #
        # print('np.array(pd.read_csv(y_dir, header=None), dtype=np.int)', np.array(pd.read_csv(y_dir, header=None), dtype=np.int))
        #
        # print('np.array(pd.read_csv(y_dir, header=None), dtype=np.int).squeeze(-1)', np.array(pd.read_csv(y_dir, header=None), dtype=np.int).squeeze(-1))

    def __getitem__(self, index):
        return self.x[index], self.y[index]  # batch_size

    def __len__(self):
        ###print()
        # print('x.shape[0]', self.x.shape[0])
        return self.x.shape[0] # sample size

# if test data has label, the TestDataset2 is not used
class TestDataset2(Dataset): # 此处对于test 集合, 因为是500个sample, 所以手动添加了12个0, sample size 500 -> 512
    def __init__(self, x_dir):
        self.x = np.array(pd.read_csv(x_dir, header=None), dtype=np.float32)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]

###print
# data = TestDataset('../2_data/test_sequence.csv', '../4_result/test_label.csv')
#
# print('data111', data.__len__())
#
# data = TestDataset('../4_result/train_para.csv', '../4_result/train_label.csv')
#
# print('data222', data.__len__())
#
# data = TestDataset('../4_result/val_para.csv', '../4_result/val_label.csv')
#
# print('data333', data.__len__())
#
# data = TestDataset2('../2_data/test_sequence.csv')
#
# print('data444', data.__len__())


