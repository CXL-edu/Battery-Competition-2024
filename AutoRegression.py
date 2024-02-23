""" 单变量时序自回归 """

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from models.DLinear import Model

id_trajectory = [12, 13]
template = 'M0%02d'

input_len, pred_len = 10, 10

class BatteryDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return x, y


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return None
        # raise AttributeError(f"'AttrDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


data = []
for id in id_trajectory:
    battery_id = template % id
    data_temp = pd.read_csv(f'train/{battery_id}_工况状态.csv', header=0)['放电容量/Ah'].values[2:-1]
    data_temp = BatteryDataset(data_temp, input_len, pred_len)
    data.append(data_temp)
dataset = ConcatDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)



configs = AttrDict(dict({'seq_len': input_len, 'pred_len': pred_len, 'in_dim': 1, 'out_dim': 1, 'individual': False, 'model_name': 'DLinear'}))
model = Model(configs)


""" 训练模型 """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).float()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = inputs.unsqueeze(-1).float(), labels.unsqueeze(-1).float()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader):.3f}')
print('Finished Training')

""" 保存模型 """
torch.save(model.state_dict(), 'seq2seq.pth')
print('Model saved')


""" 加载模型 """
# configs = AttrDict(dict({'seq_len': input_len, 'pred_len': pred_len, 'in_dim': 1, 'out_dim': 1, 'individual': False, 'model_name': 'DLinear'}))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Model(configs)
# model.load_state_dict(torch.load('seq2seq.pth'))
# model.to(device)
# model.eval()


""" M005号电池的后面一部分数据没有工况时间，故使用容量进行自回归预测 """
# data = pd.read_csv('test1/M005_res.csv', header=0)
# data_temp = data['放电容量/Ah'].dropna()
# pred_lens = len(data) - len(data_temp)
# input = torch.tensor(data_temp.values[-10:]).unsqueeze(-1).float().to(device)
# input = input.unsqueeze(0)
#
# y_pred = []
# count = 0
# while pred_lens > count:
#     output = model(input)
#     input = output
#     y_pred.append(output)
#     count += pred_len
#
# y_pred = torch.cat(y_pred, dim=1).squeeze(0).cpu().detach().numpy()
# y_pred = y_pred.reshape(-1, )[:pred_lens]
# # 保存结果到csv文件
# # data['放电容量/Ah'][-pred_lens:] = y_pred
# data.loc[data.index[-pred_lens:], '放电容量/Ah'] = y_pred
# data.to_csv('M005_.csv', index=False)


""" 替换M007号电池的部分预测结果 """
# data = pd.read_csv('result.csv', header=0)
# data_temp = data[data['电池编号'] == 'M007']
# pred_lens = len(data_temp) - 390
# input = torch.tensor(data_temp['放电容量/Ah'].iloc[380:390].values).unsqueeze(-1).float().to(device)
# input = input.unsqueeze(0)
#
# y_pred = []
# count = 0
# while pred_lens > count:
#     output = model(input)
#     input = output
#     y_pred.append(output)
#     count += pred_len
#
# y_pred = torch.cat(y_pred, dim=1).squeeze(0).cpu().detach().numpy()
# y_pred = y_pred.reshape(-1, )[:pred_lens]
# # 保存结果到csv文件
# data_temp.loc[data_temp.index[-pred_lens:], '放电容量/Ah'] = y_pred
# data[data['电池编号'] == 'M007'] = data_temp
# data.to_csv('result_replace_M07.csv', index=False)


""" 替换M015号电池的部分预测结果 """
# data = pd.read_csv('result_replace_M07.csv', header=0)
# data_temp = data[data['电池编号'] == 'M015']
# pred_lens = len(data_temp) - 140
# input = torch.tensor(data_temp['放电容量/Ah'].iloc[130:140].values).unsqueeze(-1).float().to(device)
# input = input.unsqueeze(0)
#
# y_pred = []
# count = 0
# while pred_lens > count:
#     output = model(input)
#     input = output
#     y_pred.append(output)
#     count += pred_len
#
# y_pred = torch.cat(y_pred, dim=1).squeeze(0).cpu().detach().numpy()
# y_pred = y_pred.reshape(-1, )[:pred_lens]
# # 保存结果到csv文件
# data_temp.loc[data_temp.index[-pred_lens:], '放电容量/Ah'] = y_pred
# data[data['电池编号'] == 'M015'] = data_temp
# data.to_csv('result_replace_M07_M15_.csv', index=False)




