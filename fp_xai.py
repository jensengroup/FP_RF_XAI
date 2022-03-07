
import pickle
import sys

import pandas as pd
import numpy as np

import scipy.stats as ss

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps

from copy import deepcopy

from scipy.stats import pearsonr


def mol2fp(mol, radius=2, n_bits=1024):
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bi)
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(int)


def mod_fp(mol, atom_idx, radius=2, n_bits=1024):
    
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)
    
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0: # for radius 0
                bitmap[at1][bit] = 0
            else: # for radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0
    
    new_fp = fps_morgan2 & bitmap[atom_idx]
    return fps_morgan2, new_fp, info
                    

def mod_fp_exchange(mol, idx_atom, radius=2, dummy_atom_no=47, n_bits=1024):
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=info)
    
    mol_cpy = deepcopy(mol)
    mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
    
    new_info = {}
    new_fp = AllChem.GetMorganFingerprintAsBitVect(mol_cpy, radius, nBits=n_bits, bitInfo=new_info)
    
    return fp, new_fp, info, new_info
    

def ML_normalized_weights(mol, radius=2, n_bits=1024): 
    weights = []
    uncertainty_weights = []
  
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)

    tree_predictions = []
    for tree in m.estimators_:
        tree_predictions.append(tree.predict(np.array([list(fps_morgan2)]))[0])

    orig_uq = np.std(np.array(tree_predictions))
    orig_pp = m.predict(np.array([list(fps_morgan2)]))[0]
  
    # get bits for each atom
  
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0: # for radius 0
                bitmap[at1][bit] = 0
            else: # for radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0
    # loop over atoms
    for at1 in range(mol.GetNumAtoms()):
        new_fp = fps_morgan2 & bitmap[at1]
        new_pp = m.predict(np.array([list(new_fp)]))[0]
        weights.append(orig_pp-new_pp)
        tree_predictions = []
        for tree in m.estimators_:
            tree_predictions.append(tree.predict(np.array([list(new_fp)]))[0])
        

        new_uq = np.std(np.array(tree_predictions))
        uncertainty_weights.append(orig_uq-new_uq)
  
    Normalized_weightsML = weights/np.linalg.norm(weights)
    return Normalized_weightsML.flatten(), uncertainty_weights, orig_uq


def RDKit_normalized_weights(mol):
    contribs_update = []
    mol = Chem.AddHs(mol)
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    contribs = [x for x,y in contribs]
    #molecule = mols
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1: 
            index = atom.GetIdx()
            for nabo in atom.GetNeighbors():
                index_nabo= nabo.GetIdx() 
                contribs[index_nabo] += contribs[index]

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            index_update = atom.GetIdx()
            contribs_update.append(contribs[index_update]) 
    contribs_update = np.array(contribs_update)
    Normalized_weightsRDkit = contribs_update.flatten()/np.linalg.norm(contribs_update.flatten())
    return contribs_update, Normalized_weightsRDkit 


def RDKit_bit_vector(mol, model, radius=2, n_bits=1024):
    rdkit_contrib, _ = RDKit_normalized_weights(mol)
    rdkit_bit_contribs = []
    
    ML_weights = []
  
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)


    orig_pp = model.predict(np.array([list(fps_morgan2)]))[0]
  
    # get bits for each atom
  
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0: # for radius 0
                bitmap[at1][bit] = 0
            else: # for radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0
    # loop over atoms
    for at1 in range(mol.GetNumAtoms()):
        new_fp = fps_morgan2 & bitmap[at1]
        new_pp = model.predict(np.array([list(new_fp)]))[0]
        ML_weights.append(orig_pp-new_pp) 
        removed_bits = []
        for x in fps_morgan2.GetOnBits():
            if x not in new_fp.GetOnBits():
                removed_bits.append(x)
        bit_contrib = 0
        for bit in removed_bits:
            for at, rad in info[bit]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bit_contrib += rdkit_contrib[at2]
        rdkit_bit_contribs.append(bit_contrib)
    
    return np.array(ML_weights), np.array(rdkit_contrib), np.array(rdkit_bit_contribs)


def uncertainty_vector(mol, radius=2, n_bits=1024):

    
    weights = []
    sigmas = []
  
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)

    tree_predictions_og = []
    for tree in m.estimators_:
        tree_predictions_og.append(tree.predict(np.array([list(fps_morgan2)])))
    
    orig_pp = m.predict(np.array([list(fps_morgan2)]))[0]
  
    # get bits for each atom
  
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0: # for radius 0
                bitmap[at1][bit] = 0
            else: # for radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0
    # loop over atoms
    for at1 in range(mol.GetNumAtoms()):
        new_fp = fps_morgan2 & bitmap[at1]
        tree_predictions = []
        for tree in m.estimators_:
            tree_predictions.append(tree.predict(np.array([list(new_fp)])))

        weight_trees = np.array(tree_predictions_og)-np.array(tree_predictions)
        median_weight = np.median(weight_trees)
        sigma_weight = np.std(weight_trees)
        #UQ_new = np.std(np.array(tree_predictions), axis=0)
        #new_pp = m.predict(np.array([list(new_fp)]))[0]
        #weights.append(UQ-UQ_new) 
        weights.append(median_weight)
        sigmas.append(sigma_weight)
        

    #Normalized_weightsML = weights/np.linalg.norm(weights)
    return np.array(weights).flatten(), np.array(sigmas).flatten()


def get_weights_for_visualization(mol, model, radius=2, n_bits=1024):

    ml_weights, atom_weights, FPA_weights = RDKit_bit_vector(mol, model, radius=radius, n_bits=n_bits)
    
    #atom_weights, _ = RDKit_normalized_weights(mol)
    clogp = Chem.Crippen.MolLogP(mol)
    fp = mol2fp(mol, radius=2, n_bits=2048)
    logp_pred = model.predict([fp])[0]
    #print(clogp, logp_pred)
    if np.sign(np.sum(ml_weights)) != np.sign(logp_pred):
        print("ml weights: sign problem detected")
        #ml_weights = -ml_weights
    if np.sign(np.sum(FPA_weights)) != np.sign(clogp):
        print("FPA weights: sign problem detected") 
    
    ml_weights = ml_weights*abs(logp_pred/np.sum(ml_weights))
    FPA_weights = FPA_weights*abs(clogp/np.sum(FPA_weights))

    return ml_weights, atom_weights, FPA_weights


def get_contour_image(mol, weights, contour_step=0.06):
    N_contours = (max(weights)-min(weights))/contour_step

    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, contourLines=round(N_contours))

    return fig

if __name__ == "__main__":
    model_filename = sys.argv[1]
    m = pickle.load(open(model_filename, 'rb'))
    smiles = "CC[NH+](CC)[C@@H]1CCN(C(=O)N[C@@H]2CCCC2)C1"
    mol = Chem.MolFromSmiles(smiles)
    ml_weights, atom_weights, fpa_weights = get_weights_for_visualization(mol, m, radius=2, n_bits=2048)
    fig = get_contour_image(mol, ml_weights)
    fig.savefig("ml_weights_example.pdf", bbox_inches='tight')

