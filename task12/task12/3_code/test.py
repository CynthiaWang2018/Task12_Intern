# This file is to test
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Test1Model
from dataset import TestDataset, TestDataset2
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

params = {
    'num_epochs': 2,
    'batch_size': 32,
    'lr': 0.001
}
test_dataset = TestDataset2('../4_result/test_para.csv')
test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Test1Model(device, input_size=12, hidden_size=[32, 256], num_classes=12, num_layer=3).to(device)
model.load_state_dict(torch.load('model12_0623_1.ckpt'))
total_step = len(test_dataloader)
###print()
print('total_step2', total_step)

def metrics_fn(y_pred, y_true):
    y_true = y_true.contiguous()
    y_pred = F.softmax(y_pred)
    y_pred = torch.max(y_pred, 1)[1]
    return torch.mean(y_true.view(-1).eq(y_pred).float(), 0)

result = []
with torch.no_grad():
    # total = 0
    # correct = 0
    for x in test_dataloader:
        x = x.to(device)
        outputs = model(x)
        y_pred = F.softmax(outputs)
        y_pred = torch.max(y_pred, 1)[1]
        result.extend(y_pred.cpu().detach().numpy())

df = pd.DataFrame(result)
df.to_csv('submission_0623_2.csv', header=None, index=None)
print('Finished Testing')