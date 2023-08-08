import dgl
import math
import torch as th
import numpy as np

#nomarlize
def dic_normalize(dic):
    #print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    #print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

#three letter symbol of amino acid
res_dict ={'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',
           'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    #Maps inputs not in the allowable set to the last element.
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    #print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

'''
def TargetToGraph(contact_matrix, seq3, contact=1):

    c_size = len(contact_matrix)
    G = dgl.DGLGraph()
    G.add_nodes(c_size)

    seq = [res_dict[chars] if chars in res_dict.keys() else 'X' for chars in seq3]
    node_features = seq_feature(seq)
    #edge_features = []
    for i in range(len(contact_matrix)):
        #atom_i = mol.GetAtomWithIdx(i)
        #residue_i_features = features[i]
        #node_features.append(residue_i_features)

        for j in range(len(contact_matrix)):
            contact_ij = contact_matrix[i][j]
            if i!=j and contact_ij==contact:
                G.add_edges(i, j)
                #bond_features_ij = GetBondFeatures(bond_ij)
                #edge_features.append(bond_features_ij)
    G.ndata['x'] = torch.from_numpy(np.array(node_features))  # dgl添加氨基酸残基/节点特征
    #G.edata['w'] = torch.from_numpy(np.array(edge_features)) # dgl添加contact/边特征
    return G
'''


#calculate cosine similarity
def cos_sim(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))

    #range [-1,1]
    return cos_sim


#calculate angle between two vectors
def cal_angle(point_a, point_b, point_c):
    """
    根据三点坐标计算夹角

                  点a
           点b ∠
                   点c

    :param point_a、point_b、point_c: 数据类型为list,二维坐标形式[x、y]或三维坐标形式[x、y、z]
    :return: 返回角点b的夹角值

    数学原理：
    设m,n是两个不为0的向量，它们的夹角为<m,n> (或用α ,β,θ,..,字母表示)

    1 由向量公式：cos<m,n>=m.n/|m||n|
    2 若向量用坐标表示，m=(x1,y1,z1), n=(x2,y2,z2),
    则,m.n=(x1x2+y1y2+z1z2).
    |m|=√(x1^2+y1^2+z1^2), |n|=√(x2^2+y2^2+z2^2).

    将这些代入②得到：
    cos<m,n>=(x1x2+y1y2+z1z2)/[√(x1^2+y1^2+z1^2)*√(x2^2+y2^2+z2^2)]
    上述公式是以空间三维坐标给出的，令坐标中的z=0,则得平面向量的计算公式。
    两个向量夹角的取值范围是：[0,π].
    夹角为锐角时, cosθ>0; 夹角为钝角时, cosθ<0.
    """
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        #print("坐标点为3维坐标形式")
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2] #点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0, 0, 0 #坐标点为2维坐标形式，z 坐标默认值设为0
        #print("坐标点为2维坐标形式，z 坐标默认值设为0")

    #向量m=(x1,y1,z1), n=(x2,y2,z2)
    x1, y1, z1 = (a_x-b_x), (a_y-b_y), (a_z-b_z)
    x2, y2, z2 = (c_x-b_x), (c_y-b_y), (c_z-b_z)

    #两个向量的夹角，即角点b的夹角余弦值，range [-1,1]
    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) * (math.sqrt(x2**2 + y2**2 + z2**2)))
    B = math.degrees(math.acos(cos_b)) #角点b的夹角值
    return cos_b


def TargetToGraph(contact_matrix, distance_matrix, ca_coords, seq3, contact=1, dis_min=1):
    c_size = len(contact_matrix)
    G = dgl.DGLGraph()
    G.add_nodes(c_size)

    seq = [res_dict[chars] if chars in res_dict.keys() else 'X' for chars in seq3]
    node_features = seq_feature(seq)
    edge_features = []

    for i in range(len(contact_matrix)):
        for j in range(len(contact_matrix)):
            contact_ij = contact_matrix[i][j]
            if i!=j and contact_ij==contact:
                G.add_edges(i, j)
                sim_ij = cos_sim(node_features[i], node_features[j]) #[0, 1]

                if distance_matrix[i][j] <= dis_min:
                    dis_ij = dis_min #dis_ij=1 when distance_matrix[i][j]<=1
                else:
                    dis_ij = 1 / distance_matrix[i][j] #[1/8, 1]

                angle_ij = cal_angle(ca_coords[i], [0, 0, 0], ca_coords[j]) #[-1, 1]
                contact_features_ij = [sim_ij, dis_ij, angle_ij]
                edge_features.append(contact_features_ij)

    G.ndata['x'] = th.from_numpy(np.array(node_features)).to(th.float32) #dgl添加氨基酸残基/节点特征
    G.edata['w'] = th.from_numpy(np.array(edge_features)).to(th.float32) #dgl添加contact/边特征

    return G