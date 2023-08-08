import numpy as np
import torch as th
import dgl
import rdkit
from rdkit import Chem


print (rdkit.__version__)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    #Maps inputs not in the allowable set to the last element.
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


#Get features of an atom (one-hot encoding: 元素、杂化类型、原子参与成键数:度、H离子数、隐式H离子数、是否在环上、是否芳环)
'''1.atom element: 44 dimensions (43种+其它:Unknown)       2.the atom's hybridization: 4 dimensions
   3.degree of atom: 6 dimensions                         4.total number of H bound to atom: 6 dimensions
   5.number of implicit H bound to atom: 6 dimensions     6.whether the atom is on ring: 1 dimension
   7.whether the atom is aromatic: 1 dimension            Total: 68 dimensions'''

AtomTable = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
             'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
             'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
             'Pt', 'Hg', 'Pb', 'Unknown'] #Unknown表示其它元素

def GetAtomFeatures(atom):

    '''print(atom.GetSymbol())
    print(atom.GetHybridization())
    print(atom.GetDegree())
    print(atom.GetTotalNumHs())
    print(atom.GetImplicitValence())
    print(atom.IsInRing())
    print(atom.GetIsAromatic())
    print('\n')'''

    AllFeatures = np.array(one_of_k_encoding_unk(atom.GetSymbol(), AtomTable)
                           + one_of_k_encoding(atom.GetHybridization(), [Chem.rdchem.HybridizationType.S,
                                                                         Chem.rdchem.HybridizationType.SP,
                                                                         Chem.rdchem.HybridizationType.SP2,
                                                                         Chem.rdchem.HybridizationType.SP3,
                                                                         Chem.rdchem.HybridizationType.SP3D])
                           + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                           + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
                           + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                           + [atom.IsInRing()]
                           + [atom.GetIsAromatic()])
                           # + one_of_k_encoding(atom.GetNumRadicalElectrons(), [0, 1])
                           # + one_of_k_encoding(atom.GetFormalCharge(), [-1, 0, 1])
    OneHot_AllFeatures = [int(feature) for feature in AllFeatures]

    return OneHot_AllFeatures


#Get features of an edge (one-hot encoding: 单键/双键/三键/成环、芳香环、共轭:
#π-π共轭:指两个及以上双键(或三键)以单键相联结时所发生的电子的离位作用)
'''1.single/double/triple/aromatic: 4 dimensions        2.the atom's hybridization: 1 dimensions
   3.whether the bond is on ring: 1 dimension           Total: 6 dimensions'''

def GetBondFeatures(bond):

    #print(bond.GetBondType())
    #print(bond.GetIsConjugated())
    #print(bond.IsInRing())
    #print('\n')

    bond_type = bond.GetBondType()
    BondFeatures = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    OneHot_BondFeatures = [int(feature) for feature in BondFeatures]

    return np.array(OneHot_BondFeatures)


def SmileToGraph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    G = dgl.DGLGraph()
    G.add_nodes(c_size)

    node_features = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        atom_i_features = GetAtomFeatures(atom_i)
        node_features.append(atom_i_features)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i, j)
                bond_features_ij = GetBondFeatures(bond_ij)
                edge_features.append(bond_features_ij)

    G.ndata['x'] = th.from_numpy(np.array(node_features)).to(th.float32) #dgl添加原子/节点特征
    G.edata['w'] = th.from_numpy(np.array(edge_features)).to(th.float32) #dgl添加键/边特征

    return G