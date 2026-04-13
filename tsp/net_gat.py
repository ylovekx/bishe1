import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import torch_geometric.nn as gnn


# ==========================================
# 🌟 毕设创新点：基于 GATv2 多头动态注意力机制的特征提取网络
# ==========================================
class EmbNet(nn.Module):
    def __init__(self, depth=4, feats=2, units=32, act_fn='silu'):
        super().__init__()
        self.depth = depth  # GNN 层数（12层）
        self.feats = feats  # 节点特征维度（TSP 中是 2，即坐标）
        self.units = units  # 隐藏层嵌入维度（32）
        self.act_fn = getattr(F, act_fn)

        # 1. 节点和边的特征初始化映射 (2 -> 32)
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(1, self.units)

        # -----------------------------------------------------
        # 🚀 核心创新 1：引入 GATv2Conv 多头动态图注意力层
        # heads=4 表示 4 个注意力头，每个头维度 units//4=8，拼接后依然是 32 维
        # edge_dim=units 允许注意力机制在计算权重时，把“边”的特征也考虑进去
        # -----------------------------------------------------
        self.gat_layers = nn.ModuleList([
            gnn.GATv2Conv(
                in_channels=self.units,
                out_channels=self.units // 4,
                heads=4,
                edge_dim=self.units,
                concat=True
            ) for _ in range(self.depth)
        ])

        # -----------------------------------------------------
        # 🚀 核心创新 2：边特征更新多层感知机 (Edge MLP)
        # 因为 TSP 最终需要输出边的概率，我们必须让边特征也随着节点一起进化
        # -----------------------------------------------------
        self.edge_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.units * 2 + self.units, self.units),
                nn.SiLU(),
                nn.Linear(self.units, self.units)
            ) for _ in range(self.depth)
        ])

        # 节点和边的归一化层 (防止深层网络的梯度消失/爆炸)
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for _ in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for _ in range(self.depth)])

    def forward(self, x, edge_index, edge_attr):
        # 初始特征提取
        x = self.act_fn(self.v_lin0(x))
        w = self.act_fn(self.e_lin0(edge_attr))

        # 深层特征传播
        for i in range(self.depth):
            x0 = x
            w0 = w

            # --- 步骤 A：利用 GATv2 更新节点特征 ---
            # GAT 会自动根据 edge_index 和 当前边特征 w，计算动态注意力权重
            x_new = self.gat_layers[i](x, edge_index, edge_attr=w)
            x = x0 + self.act_fn(self.v_bns[i](x_new))  # 残差连接

            # --- 步骤 B：融合节点特征，更新边特征 ---
            row, col = edge_index
            # 把边两端的节点特征(x[row], x[col])与边本身特征(w0)拼接，维度变为 32+32+32=96
            edge_cat = torch.cat([x[row], x[col], w0], dim=-1)
            w_new = self.edge_mlp[i](edge_cat)
            w = w0 + self.act_fn(self.e_bns[i](w_new))  # 残差连接

        return w  # 输出 [n_edges, 32]，无缝衔接下游 ParNet


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x)
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_phe = ParNet()
        self.par_net_heu = ParNet()

    def forward(self, pyg):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix