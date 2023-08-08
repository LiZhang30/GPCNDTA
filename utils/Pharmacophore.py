import os
import copy
import rdkit
import pprint
import subprocess
from rdkit import Chem
from IPython import display
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem import PyMol
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdShapeHelpers
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole

#RDkit version no.
print(rdkit.__version__)

#使用特征工厂搜索特征
#搜索到的每个特征都包含了: 特征家族(e.g., 受体、供体等)、特征类别、特征对应的原子、特征对应的序号等
'''for f in feats:
    print(
        f.GetFamily(),  # 特征家族信息
        f.GetType(),    # 特征类型信息
        f.GetAtomIds()  # 特征对应原子
    )
'''

#get pharmacophore vector according to smiles
fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
keys_list = list(fdef.GetFeatureDefs().keys())
'''print(keys_list)
print(len(keys_list))'''

def  get_pharmacophore(smiles_list):
    pharma = []
    for i in range(len(smiles_list)):
        mol = Chem.MolFromSmiles(smiles_list[i])
        mol_feats = fdef.GetFeaturesForMol(mol)

        buff = [0]*27
        for feat in mol_feats:
            #print(feat.GetFamily(), feat.GetType(), feat.GetAtomIds())
            #print(pos.x, pos.y, pos.z)
            feat_FT = feat.GetFamily() + '.' + feat.GetType()
            if feat_FT in keys_list:
                index = keys_list.index(feat_FT)
                buff[index] = buff[index] + 1
        #print(buff)
        pharma.append(buff)

    return pharma

'''
#transfer drug SMILE to drug 3D molecular coordinates
smiles = 'CC(C)CC(C(=O)NC(CCC(=O)N)C(=O)C(F)(F)F)NC(=O)C(C(C)C)NC(=O)C(C)NC(=O)OCC1=CC=CC=C1'
mol = AllChem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)
Chem.MolToMolFile(mol,'./before_align.sdf')


#draw molecular structure
mols = Chem.SDMolSupplier('D:/zl/GNN-DTA/utils/before_align.sdf', removeHs=False)
cdk2mol = [m for m in mols]
#print(cdk2mol)
for m in cdk2mol:
    AllChem.EmbedMolecule(m, AllChem.ETKDGv2())
img= Draw.MolsToGridImage(cdk2mol)
img.save('drug.png')


#highlight pharmacophore
cdk2mol2 = copy.deepcopy(cdk2mol)
crippen_contribs = [rdMolDescriptors._CalcCrippenContribs(mol) for mol in cdk2mol2]
ref = cdk2mol2[0]
ref_contrib = crippen_contribs[0]
targets = cdk2mol2[1:]
targets_contrib = crippen_contribs[1:]

for i, target in enumerate(targets):
    crippenO3A = rdMolAlign.GetCrippenO3A(target, ref, targets_contrib[i], ref_contrib)
    crippenO3A.Align()

w = Chem.SDWriter('align.sdf')
w.write(ref)
for mol in targets:
    w.write(mol)
w.close()


#Apply ShowFeats.py
showfeatpath = os.path.join('D://zl//GNN-DTA//rdkit-master//rdkit','Chem/Features/ShowFeats.py')
#Before align
v = PyMol.MolViewer()
v.DeleteAll()

process = subprocess.Popen(['python', showfeatpath, '--writeFeats', 'align.sdf'], stdout=subprocess.PIPE)
stdout = process.communicate()[0]
png=v.GetPNG()
display.display(png)
res = stdout.decode('utf-8').replace('\t', ' ').split('\n')
pprint.pprint(res)'''