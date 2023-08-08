import torch as th
import torch.nn as nn
import torch.nn.functional as F
from model.CensNet_layer import GraphConvolution

#best choice at present
class CensNet(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, nclass, dropout, node_layer=True):
        super(CensNet, self).__init__()

        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.fc1 = nn.Sequential(
            nn.Linear(nfeat_v, nhid, bias=False),
            nn.LayerNorm(nhid),
            nn.ReLU(inplace=True),
        )

        self.gc2 = GraphConvolution(nhid*2, nhid*2, nfeat_e, nfeat_e, node_layer=False)
        self.fc2 = nn.Sequential(
            nn.Linear(nfeat_e, nfeat_e, bias=False),
            nn.LayerNorm(nfeat_e),
            nn.ReLU(inplace=True),
        )

        self.gc3 = GraphConvolution(nhid*2, nhid, nfeat_e*2, nfeat_e*2, node_layer=True)
        self.gc4 = GraphConvolution(nhid, nhid, nfeat_e*2, nfeat_e, node_layer=False)
        self.gc5 = GraphConvolution(nhid, nclass, nfeat_e, nfeat_e, node_layer=True)
        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        #print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X1, Z1 = F.relu(gc1[0]), F.relu(gc1[1])
        X1 = F.dropout(X1, self.dropout, training=self.training)
        Z1 = F.dropout(Z1, self.dropout, training=self.training)
        F1 = self.fc1(X)
        X1F1 = th.cat((X1, F1), 1)

        gc2 = self.gc2(X1F1, Z1, adj_e, adj_v, T)
        X2, Z2 = F.relu(gc2[0]), F.relu(gc2[1])
        X2 = F.dropout(X2, self.dropout, training=self.training)
        Z2 = F.dropout(Z2, self.dropout, training=self.training)
        F2 = self.fc2(Z)
        Z2F2 = th.cat((Z2, F2), 1)

        gc3 = self.gc3(X2, Z2F2, adj_e, adj_v, T)
        X3, Z3 = F.relu(gc3[0]), F.relu(gc3[1])
        X3 = F.dropout(X3, self.dropout, training=self.training)
        Z3 = F.dropout(Z3, self.dropout, training=self.training)

        gc4 = self.gc4(X3, Z3, adj_e, adj_v, T)
        X4, Z4 = F.relu(gc4[0]), F.relu(gc4[1])
        X4 = F.dropout(X4, self.dropout, training=self.training)
        Z4 = F.dropout(Z4, self.dropout, training=self.training)

        X5, Z5 = self.gc5(X4, Z4, adj_e, adj_v,T)
        #return F.log_softmax(X, dim=1)
        return X5