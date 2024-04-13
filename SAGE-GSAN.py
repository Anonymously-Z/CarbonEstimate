import math

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
from dgl import nn
from dgl.data import DGLDataset
from dgl.nn.pytorch import SAGEConv
from sklearn import metrics
from torch import nn
from torch_geometric.data import Data

from GSAN import GraphSpatialAttentionNetwork

# 用GPU
print(torch.cuda.is_available())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

# 设置tensor的精度为小数点后precision位
torch.set_printoptions(precision=8)

torch.set_printoptions(profile="full")
# torch.set_printoptions(profile="default") # reset
# np.set_printoptions(suppress=True) //阵列显示不完整
torch.set_printoptions(sci_mode=False)

## 图构建

# 路网关系图(图的edge_index)构建
osm = pd.read_csv('Adjacency relation.csv')  # Road adjacency read
edge_index = []
O = osm['RoadID'].values.tolist()
D = osm['RoadID_2'].values.tolist()
for index in range(len(O)):
    edge_index.append([O[index] - 1, D[index] - 1])
edge_index = torch.tensor(edge_index,
                          dtype=torch.long).t()  # 用作索引的张量必须是长张量、字节张量或布尔张量

# 图的x构建
embd = pd.read_csv(
    'The semantic segmentation features of each road after processing.csv')  # 道路特征读取，一共有5075个道路
data_x = []
for i in range(5075):
    try:
        e = embd[(embd['RoadID']) == i + 1].values[0][2:21].tolist()
        data_x.append(e)
    except:
        data_x.append([0] * 19)  # 如果road没有特征，用0填充

# 图的y构建
data_y_org = pd.read_csv('CGrade.csv')  # 碳排放量读取，5075个道路
data_y = data_y_org['TotalBreak'].values.tolist()
for t in range(5075):
    if math.isnan(data_y[t]) or data_y[t] == 0:
        data_y[t] = 1  # 如果为空nan，则用0填充

# 整理数据格式
x = torch.tensor(data_x, dtype=torch.float)  # 把x转换为tensor，并设置为float浮点数，不然会报错
y = torch.tensor(data_y, dtype=torch.float)  # 把y转换为tensor，并设置为float浮点数，不然会报错

# 构建输入数据集
data = Data(x=x, edge_index=edge_index.contiguous(), y=y).to(device)  # 使用CPU计算

pos_df = pd.read_excel("FeatureToPoint.xlsx", usecols=['longitude', 'latitude'])
pos = torch.Tensor(pos_df.values.tolist())

## 构建数据集及划分数据
class RoadGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='road_network')

    def process(self):
        self.graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=data.num_nodes).to(device)
        # self.graph = dgl.add_self_loop(self.graph,fill_data='sum')

        # 需要分配掩码，指示节点是否属于训练集、验证集和测试集。
        n_nodes = data.num_nodes
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        self.graph.ndata['train_mask'] = train_mask.to(device)
        self.graph.ndata['val_mask'] = val_mask.to(device)
        self.graph.ndata['test_mask'] = test_mask.to(device)

        self.graph.ndata['feat'] = data.x
        self.graph.ndata['label'] = data.y

        self.graph.ndata['pos'] = pos

    def __getitem__(self, i):
        if i != 0:
            raise IndexError('This dataset has only one graph')
        return self.graph

    def __len__(self):
        return 1



## 构建图神经网络SAGE-GSAN
class GNN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_class):
        super(GNN, self).__init__()
        # torch.manual_seed(666666)

        self.conv1 = SAGEConv(in_feats, h_feats, 'lstm')
        self.spatial_attention = GraphSpatialAttentionNetwork(h_feats, h_feats)  # 添加空间注意力模块
        self.conv2 = SAGEConv(h_feats, num_class, 'lstm')

        # self.conv1 = GraphConv(in_feats,h_feats)
        # self.conv2 = GraphConv(h_feats,num_class)

        # self.conv1 = GCN2Conv(in_feats, layer=1, alpha=0.5,  project_initial_features=True, allow_zero_in_degree=True)
        # self.conv2 = GCN2Conv(in_feats, layer=2, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True)

        # self.conv1 = GATConv(in_feats, h_feats,4)
        # self.conv2 = GATConv(h_feats*4, num_class,1)

        # self.conv1 = GATv2Conv(in_feats, h_feats,1)
        # self.conv2 = GATv2Conv(h_feats*1, num_class,7)

        # self.conv1 = EdgeConv(in_feats, h_feats)
        # self.conv2 = EdgeConv(h_feats, num_class)

        # self.conv1 = SGConv(in_feats, h_feats,2)
        # self.conv2 = SGConv(h_feats, num_class,1)

        # self.conv1 = ChebConv(in_feats, h_feats,  2)
        # self.conv2 = ChebConv(h_feats, num_class, 2)

        # self.conv1 = TWIRLSConv(in_feats,h_feats, h_feats,prop_step = 64)
        # self.conv2 = TWIRLSConv(h_feats,num_class, num_class,prop_step = 64)

        # self.conv1 = PNAConv(in_feats, h_feats, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)
        # self.conv2 = PNAConv(h_feats, num_class,['mean', 'max', 'sum'], ['identity', 'amplification'], 2.5)

        # self.conv1 = DGNConv(in_feats, h_feats, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)
        # self.conv2 = DGNConv(h_feats, num_class, ['dir1-av', 'dir1-dx', 'sum'], ['identity', 'amplification'], 2.5)

        # self.conv1 = TAGConv(in_feats, h_feats, k=2)
        # self.conv2 = TAGConv(h_feats, num_class,2)

        self.batch_norm = nn.BatchNorm1d(h_feats)  # 添加批归一化层

    def forward(self, graph, feat, eweight=None, pos=None):
        h = self.conv1(graph, feat)
        h = F.relu(h)

        if pos is not None:
            # 如果提供了位置信息，则应用空间注意力
            h = self.spatial_attention(h, pos, edge_index)

        h = self.conv2(graph, h)
        # 在第二个卷积层后应用激活函数
        h = F.relu(h)

        graph.ndata['h'] = h
        if eweight is None:
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        else:
            graph.edata['w'] = eweight
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        return graph.ndata['h']


def train_and_pred(g, model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003,
                                 weight_decay=5e-5)  # [wd=5e-4] 0.005->0.617 || 0.004->0.627 || 0.003->0.618

    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat'].float()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # 从所有标签中减去1，因为下标从0开始
    print(len(labels[train_mask]) + len(labels[val_mask]) + len(labels[test_mask]))
    print(len(labels[train_mask]))
    print(len(labels[val_mask]))
    print(len(labels[test_mask]))

    labels[train_mask] = torch.tensor([j - 1 for j in np.array(labels[train_mask].cpu())], dtype=torch.float).to(device)
    labels[val_mask] = torch.tensor([j - 1 for j in np.array(labels[val_mask].cpu())], dtype=torch.float).to(device)
    labels[test_mask] = torch.tensor([j - 1 for j in np.array(labels[test_mask].cpu())], dtype=torch.float).to(device)

    # 训练次数
    for e in range(400):
        logits = model(g, features, pos=pos)

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask].long()).to(device)

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_test_acc < test_acc:
            best_test_acc = test_acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                'In epoch: {}, loss: {:.3f}, train_acc: {:.3f}, val_acc: {:.3f}(best {:.3f}), test_acc: {:.3f}(best {:.3f})'.format(
                    e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))

    model.eval()
    logits = model(g, features, pos=pos)
    pred = logits.argmax(1)

    pred[train_mask] = torch.tensor([j + 1 for j in np.array(pred[train_mask].cpu())], dtype=torch.long).to(device)
    pred[val_mask] = torch.tensor([j + 1 for j in np.array(pred[val_mask].cpu())], dtype=torch.long).to(device)
    pred[test_mask] = torch.tensor([j + 1 for j in np.array(pred[test_mask].cpu())], dtype=torch.long).to(device)

    labels[train_mask] = torch.tensor([j + 1 for j in np.array(labels[train_mask].cpu())], dtype=torch.float).to(device)
    labels[val_mask] = torch.tensor([j + 1 for j in np.array(labels[val_mask].cpu())], dtype=torch.float).to(device)
    labels[test_mask] = torch.tensor([j + 1 for j in np.array(labels[test_mask].cpu())], dtype=torch.float).to(device)

    # 计算每个评价指标
    MSE = metrics.mean_squared_error(pred.cpu(), labels.cpu())
    RMSE = metrics.mean_squared_error(pred.cpu(), labels.cpu()) ** 0.5
    MAE = metrics.mean_absolute_error(pred.cpu(), labels.cpu())
    MAPE = metrics.mean_absolute_percentage_error(pred.cpu(), labels.cpu())
    ME = metrics.max_error(pred.cpu(), labels.cpu())
    MSL = metrics.mean_squared_log_error(pred.cpu(), labels.cpu())
    print("MSE:{", MSE, "}RMSE:{", RMSE, "}MAE:{", MAE, "}MAPE:{", MAPE, "}ME:{", ME, "}MSL{", MSL, "}")


# 构建数据集
dataset = RoadGraphDataset()
g = dataset[0].to(device)

# 开始训练和估算
model = GNN(g.ndata['feat'].shape[1], 16, 7).to(device)
train_and_pred(g, model)
