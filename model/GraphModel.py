import dgl
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import dgl.nn.pytorch as dglnn
from model.CensNet_model import CensNet
from utils.DglOperator import BDgraph_readout

#Transformer Parameters
d_model = 128 #Embedding Size
d_ff = 512 #FeedForward dimension
d_k = d_v = 16 #dimension of K(=Q), V
n_layers = 1 #number of Encoder
n_heads = 8 #number of heads in Multi-Head Attention

class DatasetIterater(Data.Dataset):
    def __init__(self, texta, textb, label):
        self.texta = texta
        self.textb = textb
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.textb[item], self.label[item]

    def __len__(self):
        return len(self.texta)


class DatasetIteraterP(Data.Dataset):
    def __init__(self, texta):
        self.texta = texta

    def __getitem__(self, item):
        return self.texta[item]

    def __len__(self):
        return len(self.texta)


def BatchPad(batch_data, pad=0):
    texta, textb, label = list(zip(*batch_data))
    max_len_a = max([len(seq_a) for seq_a in texta])
    max_len_b = max([len(seq_b) for seq_b in textb])
    texta = [seq+[pad]*(max_len_a-len(seq)) for seq in texta]
    textb = [seq+[pad]*(max_len_b-len(seq)) for seq in textb]
    texta = th.LongTensor(texta)
    textb = th.LongTensor(textb)
    label = th.FloatTensor(label)
    return (texta, textb, label)


def Pcollate(batch_data):
    simple_array = np.array(batch_data, dtype=np.float32)
    texta = th.FloatTensor(simple_array)
    return texta


def Gcollate(samples):
    dgraphs, tgraphs, labels = map(list, zip(*samples))
    batched_dgraph = dgl.batch(dgraphs)
    batched_dgraph.ndata['x'] = th.as_tensor(batched_dgraph.ndata['x'], dtype=th.float32)
    batched_dgraph.edata['w'] = th.as_tensor(batched_dgraph.edata['w'], dtype=th.float32)

    batched_tgraph = dgl.batch(tgraphs)
    batched_tgraph.ndata['x'] = th.as_tensor(batched_tgraph.ndata['x'], dtype=th.float32)

    batched_labels = th.as_tensor(labels, dtype=th.float32)
    return batched_dgraph, batched_tgraph, batched_labels


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    #seq_q=seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) #[batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k) #[batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #Q: [batch_size, n_heads, len_q, d_k]
        #K: [batch_size, n_heads, len_k, d_k]
        #V: [batch_size, n_heads, len_v(=len_k), d_v]
        #attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = th.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) #scores:[batch_size, n_heads, len_q, len_k]

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) #Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = th.matmul(attn, V) #[batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, len_q, d_model]
        #input_K: [batch_size, len_k, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]

        #batch_size, seq_len, model_len = input_Q.size()
        if attn_mask is not None:
            batch_size, seq_len, model_len = input_Q.size()
            residual_2D = input_Q.view(batch_size*seq_len, model_len)
            residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)
        else:
            residual, batch_size = input_Q, input_Q.size(0)

        '''residual, batch_size = input_Q, input_Q.size(0)'''
        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            #attn_mask:[batch_size, n_heads, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn


class Inner_MultiHeadAttention(nn.Module):
    def __init__(self):
        super(Inner_MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, d_model]
        #input_K: [batch_size, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size, len_q = input_Q, input_Q.size(0), input_Q.size(1)

        #input_K1: [batch_size, len_k, d_model]
        #input_V1: [batch_size, len_k, d_model]
        input_K1 = input_K.unsqueeze(1).repeat(1, len_q, 1)
        input_V1 = input_V.unsqueeze(1).repeat(1, len_q, 1)

        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K1).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V1).view(batch_size, -1, n_heads, d_v).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) #attn_mask:[bs, heads, seq_len, seq_len]

        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        #inputs:[batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to('cuda:0')(output + residual) #[batch_size, seq_len, d_model]


class Inner_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inner_EncoderLayer, self).__init__()
        self.enc_self_attn = Inner_MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn


class Inter_EncoderLayer(nn.Module):
    def __init__(self):
        super(Inter_EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.stream = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        #enc_inputs:[batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs) #[batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) #[batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) #[batch_size, src_len, src_len]
        stream = enc_outputs

        enc_self_attns = []
        for layer in self.stream:
            #enc_outputs:[batch_size, src_len, d_model]
            #enc_self_attn:[batch_size, n_heads, src_len, src_len]
            stream, enc_self_attn = layer(stream, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        #skip connect -> stream
        out_stream = stream + enc_outputs
        return out_stream, enc_self_attns


#best choice at present
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim*4, bias=False)
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim*4, bias=False),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(inplace=True),
        )

        self.layer2 = dglnn.GraphConv(hidden_dim*8, hidden_dim*4, bias=False)
        self.layer3 = dglnn.GraphConv(hidden_dim*4, out_dim, bias=False)

    def forward(self, graph, x, w):
        x1 = self.layer1(graph, x, edge_weight=w)
        x1 = F.relu(x1, inplace=True)
        f1 = self.fc1(x)
        x1f1 = th.cat((x1, f1), 1)

        x2 = self.layer2(graph, x1f1, edge_weight=w)
        x2 = F.relu(x2, inplace=True)

        x3 = self.layer3(graph, x2, edge_weight=w)
        x3 = F.relu(x3, inplace=True)

        with graph.local_scope():
            graph.ndata['x'] = x3
            readout = dgl.sum_nodes(graph, 'x')

            return readout


class GraphBasedModel(nn.Module):
    def __init__(self, model_config):
        super(GraphBasedModel, self).__init__()
        #model config
        self.model_config = model_config
        self.din_dim = model_config['drug_node_dim']
        self.die_dim = model_config['drug_edge_dim']
        self.tin_dim = model_config['target_node_dim']
        self.tie_dim = model_config['target_edge_dim']
        self.pin_dim = model_config['pharmacophore_dim']
        self.hidden_dim = model_config['node_hidden_dim']
        self.out_dim = model_config['node_out_dim']
        self.dru_dic = model_config['drug_vocab']
        self.tar_dic =model_config['target_vocab']
        self.dropout_ratio = model_config['dropout_ratio']
        self.reg_dim = model_config['regression_value_dim']

        #model layers
        self.encoderD = Encoder(self.dru_dic)
        self.encoderT = Encoder(self.tar_dic)

        self.dCensNet = CensNet(self.din_dim, self.die_dim, self.hidden_dim, self.out_dim, self.dropout_ratio)
        self.trans_pharma_dim = nn.Sequential(
            nn.Linear(self.pin_dim, self.hidden_dim*4, bias=False),
            nn.LayerNorm(self.hidden_dim*4),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim*4, self.hidden_dim, bias=False),
        )
        self.tgcn = GCN(self.tin_dim, self.hidden_dim, self.out_dim)

        #share weights
        self.inner_cross_atten = Inner_EncoderLayer()
        self.inter_cross_atten = Inter_EncoderLayer()

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*8, bias=False),
            nn.LayerNorm(self.hidden_dim*8),
            nn.Dropout(self.dropout_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim*8, self.hidden_dim*2, bias=False),
        )

        self.reg_fun = nn.Sequential(
            nn.Linear(self.hidden_dim*4, self.hidden_dim*16, bias=False),
            nn.LayerNorm(self.hidden_dim*16),
            nn.Dropout(self.dropout_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim*16, self.hidden_dim*4, bias=False),
            nn.LayerNorm(self.hidden_dim*4),
            nn.Dropout(self.dropout_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim*4, self.reg_dim, bias=False)
        )

    def forward(self, dg, d_features, de_features, de_adj, d_adj, d_mat, pd, tg, tx, tw, ds, ts, bs):
        #token and position embedding + self-attention for drug seq and target seq (add pad mask)
        EncDru, attnsD = self.encoderD(ds)
        EncTar, attnsT = self.encoderT(ts)

        #encode and extarct graph information for drug and target
        Gdru = self.dCensNet(d_features, de_features, de_adj, d_adj, d_mat)
        readout_batchGdru = BDgraph_readout(bs, dg, Gdru)
        pharma = self.trans_pharma_dim(pd)
        Gdru_mix_pd = readout_batchGdru + pharma #element plus: add 3D pharma to drug graph
        Gtar = self.tgcn(tg, tx, tw)

        #inner_cross_atten based on graph information for drug and target
        InnerAtten_outD, _ = self.inner_cross_atten(EncDru, Gdru_mix_pd, Gdru_mix_pd, None)
        InnerAtten_outT, _ = self.inner_cross_atten(EncTar, Gtar, Gtar, None)

        #inter_cross_atten for drug and target
        T2D_out, _ = self.inter_cross_atten(InnerAtten_outD, InnerAtten_outT, InnerAtten_outT, None)
        D2T_out, _ = self.inter_cross_atten(InnerAtten_outT, InnerAtten_outD, InnerAtten_outD, None)

        #seq features plus graph features
        din = th.cat((th.sum(T2D_out, 1), Gdru_mix_pd), 1)
        tin = th.cat((th.sum(D2T_out, 1), Gtar), 1)
        #linear projection for drug and target
        dout = self.projection(din)
        tout = self.projection(tin)
        DT_out = th.cat((dout, tout), 1)

        affi = self.reg_fun(DT_out)
        return affi