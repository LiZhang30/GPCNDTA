import dgl
import math
import json
import pickle
import numpy as np
import torch as th
import scipy.sparse as sp
from collections import OrderedDict
from utils import DrugGraph as DG
from utils import TargetGraph as TG
from dgl.data.utils import save_graphs

#pad: 0 -> Symbol that will fill in blank sequence if current batch data size is short than time steps
#vocab -> from DeepDTA

#'B''O''U''X'''Z'
targetSeq_vocab = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
                   "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
                   "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
                   "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
tar_dic = 25

#vocab -> from DeepDTA
#Canonical SMILE
drugSeq_vocab = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			      ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			      "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			      "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			      "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			      "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			      "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			      "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			      "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			      "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			      "t": 61, "y": 62}
dru_dic = 62

'''
# Iso SMILE
CharIsoSmiSet = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33,
                 "/": 34, ".": 2, "1": 35, "0": 3, "3": 36, "2": 4,
                 "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7,
                 "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46,
                 "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49,
                 "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52,
                 "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59,
                 "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62,
                 "r": 27, "u": 63, "t": 28, "y": 64}
CharIsoSmiLen = 64
'''
'''
#vocab -> from DeepAffinity
DrugDB_vocab = {"C": 1, "=": 2, "(": 3, ")": 4, "O": 5, "N": 6, "1": 7, "2": 8, "3": 9,
              "4": 10, "[": 11, "]": 12, "S": 13, "l": 14, "F": 15, "-": 16, "5": 17,
              "+": 18, ".": 19, "6": 20, "B": 21, "r": 22, "#": 23, "P": 24, "i": 25,
              "H": 26, "7": 27, "I": 28, "8": 29, "9": 30, "a": 31, "e": 32, "A": 33,
              "n": 34, "s": 35, "u": 36, "g": 37, "o": 38, "t": 39, "T": 40, "M": 41,
              "Z": 42, "b": 43, "K": 44, "R": 45, "d": 46, "W": 47, "G": 48, "L": 49,
              "c": 50, "h": 51, "V": 52, "m": 53, "E": 54, "Y": 55, "U": 56, "f": 57,
              "D": 58, "y": 59, "%": 60, "0": 61, "p": 62, "k": 63, "X": 64}
drugSeq_vocabSize = 64

targetSps_vocab = {"_PAD": 0, "CEDS": 1, "CEKS": 2, "CETS": 3, "BNGS": 4, "AEDM": 5, "CEDM": 6,
              "CEDL": 7, "AEKM": 8, "CEGS": 9, "CEKM": 10, "CETL": 11, "CETM": 12,
              "AEDL": 13, "AEKL": 14, "CEKL": 15, "ANGL": 16, "BNDS": 17, "BNTS": 18,
              "BNGM": 19, "ANGM": 20, "AETM": 21, "CEGM": 22, "AEDS": 23, "BNKS": 24,
              "CNGS": 25, "BEDS": 26, "AEGM": 27, "BNTM": 28, "AETL": 29, "CEGL": 30,
              "CNDS": 31, "ANTM": 32, "ANKM": 33, "ANDM": 34, "BNKM": 35, "CNTS": 36,
              "BEKS": 37, "BEKM": 38, "ANTL": 39, "BETS": 40, "AEKS": 41, "ANKL": 42,
              "BEDM": 43, "BNDM": 44, "CNGM": 45, "BETM": 46, "AEGL": 47, "CNKS": 48,
              "CNTM": 49, "BEGS": 50, "ANDL": 51, "ANGS": 52, "AETS": 53, "BEGM": 54,
              "ANDS": 55, "CNDM": 56, "AEGS": 57, "CNTL": 58, "CNGL": 59, "CNKM": 60,
              "ANTS": 61, "CNDL": 62, "ANKS": 63, "BNGL": 64, "CNKL": 65, "BEKL": 66,
              "BEDL": 67, "BETL": 68, "BNTL": 69, "BNKL": 70, "BNDL": 71, "BEGL": 72}
sps_vocabSize = 73
'''

#transfer token -> number
def LabelDT(drug_seqs, target_seqs, drugSeq_maxlen, targetSeq_maxLen):
    label_drugSeqs, label_targetSeqs = [], []
    drugSeq_truncated, targetSeq_truncated = drugSeq_maxlen, targetSeq_maxLen
    for i in range(len(drug_seqs)):
        label_drugSeqs.append([])
        if len(drug_seqs[i]) >= drugSeq_truncated:
            for j in range(drugSeq_truncated):
                label_drug = drugSeq_vocab[drug_seqs[i][j].split()[0]]
                label_drugSeqs[i].append(label_drug)
        else:
            for j in range(len(drug_seqs[i])):
                label_drug = drugSeq_vocab[drug_seqs[i][j].split()[0]]
                label_drugSeqs[i].append(label_drug)

        label_targetSeqs.append([])
        if len(target_seqs[i]) >= targetSeq_truncated:
            for j in range(targetSeq_truncated):
                label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                label_targetSeqs[i].append(label_traget)
        else:
            for j in range(len(target_seqs[i])):
                label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                label_targetSeqs[i].append(label_traget)

    return label_drugSeqs, label_targetSeqs


#get compile + protein pairs
def GetPairs(label_drugSeqs, label_targetSeqs):
    pairs = []
    for i in range(len(label_targetSeqs)):
        drugSeq = label_drugSeqs[i]
        targetSeq = label_targetSeqs[i]
        pairs.append(drugSeq+targetSeq) # avoid ‘extend()’

    return pairs


#load davis and kiba
def LoadData(path, logspance_trans):
    print("Read %s start" % path)
    ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
    Y = pickle.load(open(path + "Y", "rb"), encoding='latin1')  # TODO: read from raw
    if logspance_trans:
        # Y = -(np.log10(Y / (math.pow(math.e, 9))))
        Y = -(np.log10(Y / (math.pow(10, 9))))

    XD = []
    XT = []
    for d in ligands.keys():
        XD.append(ligands[d])

    for t in proteins.keys():
        XT.append(proteins[t])

    return XD, XT, Y


#load davis, kiba and protein ID
def LoadData1(path, logspance_trans):
    print("Read %s start" % path)
    ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
    Y = pickle.load(open(path + "Y", "rb"), encoding='latin1')  # TODO: read from raw
    if logspance_trans:
        #Y = -(np.log10(Y / (math.pow(math.e, 9))))
        Y = -(np.log10(Y / (math.pow(10, 9))))

    XD = []
    XT = []
    XT_ID = []
    for d in ligands.keys():
        XD.append(ligands[d])

    for t in proteins.keys():
        XT.append(proteins[t])
        XT_ID.append(t)

    return XD, XT, XT_ID, Y


#load filtered_davis, metz and toxcast
def LoadData2(path, logspance_trans=None):
    drug_info = np.loadtxt(path+'drug_info.txt', dtype=str, comments='!').tolist()
    targ_info = np.loadtxt(path+'targ_info.txt', dtype=str, comments='!').tolist()
    affi_info = np.loadtxt(path+'affi_info.txt', dtype=str, comments='!').tolist()

    return drug_info, targ_info, affi_info


#create samples for davis and kiba
def GetSamples(dataSet_name, drugSeqs, targetSeqs, affi_matrix):
    drugSeqs_buff, targetSeqs_buff, affiMatrix_buff= [], [], []
    if dataSet_name == 'davis':
        for i in range(len(drugSeqs)):
            for j in range(len(targetSeqs)):
                drugSeqs_buff.append(drugSeqs[i])
                targetSeqs_buff.append(targetSeqs[j])
                affiMatrix_buff.append(affi_matrix[i, j])

    if dataSet_name == 'kiba':
        for a in range(len(drugSeqs)):
            for b in range(len(targetSeqs)):
                if  ~(np.isnan(affi_matrix[a, b])):
                    drugSeqs_buff.append(drugSeqs[a])
                    targetSeqs_buff.append(targetSeqs[b])
                    affiMatrix_buff.append(affi_matrix[a, b])

    return drugSeqs_buff, targetSeqs_buff, affiMatrix_buff


#create samples for filtered_davis, metz and toxcast
def GetSamples1(dataSet_name, drug_info, drug_graphs, target_info, taget_graphs, affi):
    gdrug_buf, gtarget_buf, gaffi_buf = [], [], []
    sdrug_buf, starget_buf, saffi_buf = [], [], []

    if dataSet_name=='filtered davis':
        for m in range(len(affi)):
            sdrug_buf.append(affi[m][1])
            starget_buf.append(affi[m][3])
            saffi_buf.append(affi[m][4])

        for i in range(len(affi)):
            for j in range(len(drug_info)):
                for k in range(len(target_info)):
                    if (affi[i][0])==drug_info[j] and (affi[i][2]==target_info[k]):
                        gdrug_buf.append(drug_graphs[j])
                        gtarget_buf.append(taget_graphs[k])
                        gaffi_buf.append(affi[i][4])

    f_gaffi_buf = [np.float32(val) for val in gaffi_buf]
    f_saffi_buf = [np.float32(val) for val in saffi_buf]

    return gdrug_buf, gtarget_buf, f_gaffi_buf, sdrug_buf, starget_buf, f_saffi_buf


#shuttle
def Shuttle(drug, target, affini, index):
    drug = np.array(drug, dtype=object)
    target = np.array(target, dtype=object)
    affini = np.array(affini, dtype=object)

    shttle_drug = drug[index]
    shttle_target = target[index]
    shttle_affini = affini[index]

    return shttle_drug, shttle_target, shttle_affini

#shuttle pharma
def Shuttle_pharma(drug, index):
    drug = np.array(drug, dtype=object)
    shttle_drug = drug[index]

    return shttle_drug


#row-normalize sparse matrix
def normalize(mx):
    mx_buff = mx.cpu().numpy() #convert to numpy

    rowsum = np.array(mx_buff.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx_buff = r_mat_inv.dot(mx_buff)

    mx_buff = th.from_numpy(mx_buff).to(th.float32)
    return  mx_buff #convert to tensor


#dense tensor to sparse tensor
def dense_to_coo(dense_tensor):
    dense_tensor_idx = th.nonzero(dense_tensor).T
    dense_tensor_data = dense_tensor[dense_tensor_idx[0], dense_tensor_idx[1]]
    coo_tensor = th.sparse_coo_tensor(dense_tensor_idx, dense_tensor_data, dense_tensor.size())

    return coo_tensor


#batch normalize features and adjacency matrix (node and edge)
def batch_normalize(batch_graph):
    #detach features, e_features, e_adj, adj, t_mat
    t_mat = th.abs(batch_graph.incidence_matrix(typestr='both').to_dense())
    sparse_t_mat = dense_to_coo(t_mat)
    adj = batch_graph.adj()
    features = batch_graph.ndata['x']
    #convert to line graph
    e_adj = dgl.line_graph(batch_graph).adj()
    e_features = batch_graph.edata['w']

    nor_features = normalize(features)
    nor_e_features = normalize(e_features)
    #adjacency matrix + identity matrix
    nor_adj = dense_to_coo(normalize(th.eye(adj.size(0))+adj))
    nor_e_adj = dense_to_coo(normalize(th.eye(e_adj.size(0))+e_adj))

    return nor_features, nor_e_features, nor_e_adj, nor_adj, sparse_t_mat


##construct graphs
#load data
#fpath_davis = 'D:/zl/GNN-DTA/data/davis/'
#fpath_kiba = 'D:/zl/GNN-DTA/data/kiba/'
#drug, target, affinity = LoadData(fpath_kiba, logspance_trans=False)
#construct graphs for drugs
'''save_graphs("D:/zl/GNN-DTA/DrugGraphs.bin", [DG.SmileToGraph(smile) for smile in drug])'''
#construct graphs for targets
'''target_graphs = []
for i in range(len(target)):
    target_map = np.loadtxt('../data/davis/davis contact map/{}.txt'.format(i), dtype=int)
    target_dis = np.loadtxt('../data/davis/davis distance map/{}.txt'.format(i), dtype=float)
    target_ca = np.loadtxt('../data/davis/davis ca coords/{}.txt'.format(i), dtype=float)
    target_seq3 = np.loadtxt('../data/davis/davis seq for contact map/{}.txt'.format(i), dtype=str)

    print(i)
    print(target_map[i])
    print(target_dis[i])
    print(target_ca[i])
    print(target_seq3[i])
    target_graph = TG.TargetToGraph(target_map, target_dis, target_ca, target_seq3)

    #print(target_graph)
    target_graphs.append(target_graph)
save_graphs("D:/zl/GNN-DTA/target_graphs.bin", target_graphs)'''
