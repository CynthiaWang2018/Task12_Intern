# This file is to train
# 可以train 后直接在这里进行test
# 可以train 后保存model 在test.py中调用

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Test1Model
from dataset import TestDataset, TestDataset2

import torch.nn as nn
from torch.autograd import Variable
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    'num_epochs': 5,
    'batch_size': 32,
    'lr': 0.001
}

train_dataset = TestDataset('../4_result/train_para.csv', '../4_result/train_label.csv')
train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

val_dataset = TestDataset('../4_result/val_para.csv', '../4_result/val_label.csv')
val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

test_dataset = TestDataset2('../4_result/test_para.csv') # 补12行0 -> 512
test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

model = Test1Model(device, input_size=12, hidden_size=[32, 256], num_classes=12, num_layer=3).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
# params_len = list(model.parametes())

optimizer = optim.Adam(model.parameters(), lr=params['lr'])
total_step = len(train_dataloader)
###print()
print('total_step', total_step)

def metrics_fn(y_pred, y_true):
    y_true = y_true.contiguous()
    y_pred = F.softmax(y_pred)
    y_pred = torch.max(y_pred, 1)[1]
    return torch.mean(y_true.view(-1).eq(y_pred).float(), 0)

for epoch in range(params['num_epochs']):
    model = model.to(device)
    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        # Forward
        outputs = model(x)
        loss = loss_fn(outputs, y)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, params['num_epochs'], i + 1, total_step, loss.item()))

# Use val to test the model
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in val_dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        y_pred = F.softmax(outputs)
        y_pred = torch.max(y_pred, 1)[1]
        total += y.size(0) # 32 + 32 + ...
        correct += (y_pred == y).sum().item()
    print('Accuracy of the network on the 3200 val examples:{}%'.format(100 * correct / total))

result = []
true = []
with torch.no_grad():
    # total = 0
    # correct = 0
    for x in test_dataloader:
        x = x.to(device)
        outputs = model(x)
        y_pred = F.softmax(outputs)
        y_pred = torch.max(y_pred, 1)[1]
        result.extend(y_pred.cpu().detach().numpy())

import pandas as pd
df = pd.DataFrame(result)
df.to_csv('submission_0623_1.csv', header=None, index=None)

torch.save(model.state_dict(), 'model12_0623_1.ckpt')

print('Finished Training')


