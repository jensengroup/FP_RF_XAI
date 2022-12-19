
# For CUDA 11.3
import torch

import kora.install.rdkit

import torch_geometric
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

import torch_geometric.nn as geom_nn

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.NN = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )


    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        x = self.NN(x)
        x = self.head(x)
        return x

class GraphLevelGNN(pl.LightningModule):

    def __init__(self, batch_size, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.MSELoss()
        self.batch_size = batch_size

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        preds = x.float()
        data.y = data.y.float()

        loss = self.loss_module(x, data.y)

        return loss, preds
    
    def predict(self, data, mode="test"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        preds = x.float()
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.0) # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, mode="val")
        self.log('val_loss', loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(batch, mode="test")
        self.log('test_loss', loss, batch_size=self.batch_size)


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, y = y_tensor))
    return data_list


def remove_GC_atom(mol, y):
    #mol = Chem.MolFromSmiles(smiles)
    n_nodes = mol.GetNumAtoms()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))

    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
        #print(atom.GetIdx(), atom.GetAtomicNum())
    
    AC = GetAdjacencyMatrix(mol)
    #print(AC)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    #print(rows)
    #print(cols)

    data_list = []
    for atom in range(n_nodes):

        AC_del = np.delete(AC, atom, 0)
        AC_del = np.delete(AC_del, atom, 1)
        
        #AC_del = AC
        
        (rows, cols) = np.nonzero(AC_del)


        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)

        X_del = np.delete(X, atom, 0)
        
        #X_del = X.copy()
        #X_del[atom, :] = np.zeros(n_node_features)

        X_del = torch.tensor(X_del, dtype = torch.float)

        y_tensor = torch.tensor(np.array([y]), dtype = torch.float)

        data_list.append(Data(x = X_del, edge_index = E, y = y_tensor))

    return data_list

if __name__ == "__main__":

    smiles = pd.read_csv('ZINC_250k.smi', names=['smiles'], header=None)
    train_smiles = smiles.smiles.iloc[:145000].to_list()
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    y_train = [Descriptors.MolLogP(mol) for mol in train_mols]
    y_train = np.array(y_train)

    data_list = remove_GC_atom(train_mols[0], y_train[0])

    data_list

    m = torch.load("Model_150k_a.p", map_location=torch.device('cpu'))
    m.eval()
    print(m.training)

    m.predict(data_list[0])

    #model.model.head = Identity()
    #model.model.NN = Identity()

    val_smiles = smiles.smiles.iloc[145000:150000].to_list()
    val_mols = [Chem.MolFromSmiles(s) for s in val_smiles]
    y_val = [Descriptors.MolLogP(mol) for mol in val_mols]
    y_val = np.array(y_val)

    x_smiles = smiles.smiles.iloc[-5000:].to_list()
    mols = [Chem.MolFromSmiles(s) for s in x_smiles]
    y = [Descriptors.MolLogP(mol) for mol in mols]
    y = np.array(y)

    from rdkit.Chem import rdMolDescriptors
    def RDkit_normalized_weights(mol):
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

    v_truth, v_truth_norm = RDkit_normalized_weights(mols[0])

    def ML_normalized_weights(mol, y_pred): 

        X_removed = remove_GC_atom(mol, y_pred)
        X_removed = torch_geometric.loader.DataLoader(dataset = X_removed, batch_size = 128)

        for (k, batch) in enumerate(X_removed):
            error, preds = mp.predict_step(batch, k)
            preds = preds.detach().numpy()
            if k==0:
                y_val_preds = preds
            else:
                y_val_preds = np.append(y_val_preds, preds, axis=0)
        weights = y_pred - y_val_preds
      
        Normalized_weightsML = weights/np.linalg.norm(weights)

        return Normalized_weightsML.flatten()

    data_list_train = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(train_smiles, y_train)
    data_list_val = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(val_smiles, y_val)
    data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)

    graph_test_loader = torch_geometric.loader.DataLoader(dataset = data_list, batch_size = 128)

    graph_train_loader = torch_geometric.loader.DataLoader(dataset = data_list_train, batch_size = 128)

    graph_val_loader = torch_geometric.loader.DataLoader(dataset = data_list_val, batch_size=128)

    #predictions
    mp = torch.load("Model_150k_a.p", map_location=torch.device('cpu'))
    mp.eval()

    for (k, batch) in enumerate(graph_val_loader):
        # compute current value of loss function via forward pass
        error, preds = mp.predict_step(batch, k)
        #print(torch.sqrt(error))
        preds = preds.detach().numpy()
        if k==0:
            y_val_preds = preds
        else:
            y_val_preds = np.append(y_val_preds, preds, axis=0)
    error_val = (y_val_preds-y_val)

    for (k, batch) in enumerate(graph_test_loader):
        #print(k)
        # compute current value of loss function via forward pass
        error, preds = mp.predict_step(batch, k)
        #print(torch.sqrt(error))
        preds = preds.detach().numpy()
        if k==0:
            y_test_preds = preds
        else:
            y_test_preds = np.append(y_test_preds, preds, axis=0)
    error_test = y_test_preds-y

    for (k, batch) in enumerate(graph_train_loader):
        #print(k)
        # compute current value of loss function via forward pass
        error, preds = mp.predict_step(batch, k)
        preds = preds.detach().numpy()
        if k == 0:
            y_train_preds = preds
        else:
            y_train_preds = np.append(y_train_preds, preds, axis=0)
        #print(torch.sqrt(error))
    error_train = y_train_preds-y_train

    import matplotlib.pyplot as plt

    plt.scatter(y, y_test_preds, s=1)

    print(np.sqrt((np.mean(error_test**2))))

    for (k, batch) in enumerate(graph_val_loader):
        #print(k)
        # compute current value of loss function via forward pass
        preds = m_latent.predict(batch, k)
        preds = preds.detach().numpy()
        if k == 0:
            val_preds = preds
        else:
            val_preds = np.append(val_preds, preds, axis=0)

    for (k, batch) in enumerate(graph_test_loader):
        #print(k)
        # compute current value of loss function via forward pass
        preds = m_latent.predict(batch, k)
        preds = preds.detach().numpy()
        if k == 0:
            test_preds = preds
        else:
            test_preds = np.append(test_preds, preds, axis=0)

    gcnn_overlaps = []
    i=0
    for mol, _y in zip(mols, y_test_preds):
        v1 = ML_normalized_weights(mol, _y)
        _, v2 = RDkit_normalized_weights(mol)
        overlap = v1@v2
        gcnn_overlaps.append(overlap)
        print(i, "overlap = ", overlap)
        i+=1

    np.mean(gcnn_overlaps)

    test_df = pd.read_pickle("test_data_2048.pkl")
    print(np.sqrt(np.mean(test_df.abs_error_150000**2)))

    test_df.columns

    !pip3 install cairosvg
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG
    import cairosvg
    import math

    from rdkit.Chem import AllChem
    from rdkit.Chem import DataStructs

    def mol2fp(mol, radius=2, n_bits=2048):
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bi)
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(int)


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


    # def RDKit_bit_vector(mol, model, radius=2, n_bits=2048):
    #    rdkit_contrib, _ = RDKit_normalized_weights(mol)
    #    rdkit_bit_contribs = []
       
    #    ML_weights = []
     
    #    info = {}
    #    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)


    #     orig_pp = model.predict(np.array([list(fps_morgan2)]))[0]
      
    #     # get bits for each atom
      
    #     bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    #     for bit, es in info.items():
    #         for at1, rad in es:
    #             if rad == 0: # for radius 0
    #                 bitmap[at1][bit] = 0
    #             else: # for radii > 0
    #                 env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
    #                 amap = {}
    #                 submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    #                 for at2 in amap.keys():
    #                     bitmap[at2][bit] = 0
        
    #     # loop over atoms
    #     for at1 in range(mol.GetNumAtoms()):
    #         new_fp = fps_morgan2 & bitmap[at1]
    #         new_pp = model.predict(np.array([list(new_fp)]))[0]
    #         ML_weights.append(orig_pp-new_pp) 
        
        
    #     return np.array(ML_weights), np.array(rdkit_contrib)

    def RDKit_bit_vector(mol, model, radius=2, n_bits=2048):
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

    def get_weights_for_visualization(mol, model, radius=2, n_bits=2048):

        ml_weights, atom_weights = RDKit_bit_vector(mol, model, radius=radius, n_bits=n_bits)
        
        #atom_weights, _ = RDKit_normalized_weights(mol)
        clogp = Chem.Crippen.MolLogP(mol)
        fp = mol2fp(mol, radius=2, n_bits=2048)
        logp_pred = model.predict([fp])[0]
        #print(clogp, logp_pred)
        if np.sign(np.sum(ml_weights)) != np.sign(logp_pred):
            print("ml weights: sign problem detected")
            #ml_weights = -ml_weights
        #if np.sign(np.sum(FPA_weights)) != np.sign(clogp):
        #    print("FPA weights: sign problem detected") 
        
        ml_weights = ml_weights*abs(logp_pred/np.sum(ml_weights))
        #FPA_weights = FPA_weights*abs(clogp/np.sum(FPA_weights))

        return ml_weights, atom_weights


    def get_contour_image(mol, weights, contour_step=0.06):
        N_contours = (max(weights)-min(weights))/contour_step

        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, contourLines=round(N_contours))

        return fig

    gcnn_rf_overlaps = []
    gcnn_fpa_overlaps = []
    i=0
    for mol, _y in zip(mols, y_test_preds):
        v1 = ML_normalized_weights(mol, _y)
        v2, _, v3 = RDKit_bit_vector(mol, m_rf)
        v2 = v2/np.linalg.norm(v2)
        v3 = v3/np.linalg.norm(v3)
        overlap1 = v1@v2
        overlap2 = v1@v3
        gcnn_rf_overlaps.append(overlap1)
        gcnn_fpa_overlaps.append(overlap2)
        print(i, "overlap = ", overlap1, overlap2, v2@v2, v1@v1, v3@v3)
        i+=1

    print(np.mean(gcnn_fpa_overlaps))
    plt.hist(gcnn_fpa_overlaps, bins=30)

    np.mean(gcnn_rf_overlaps)

    test_df.overlap_direct_150000

    example_idxs = [249071, 247354, 244843, 247424]
    example_idxs = np.array(example_idxs) - 244456

    from scipy.stats import pearsonr
    pearsonr(test_df.overlap_direct_150000, gcnn_rf_overlaps)



    from rdkit.Chem.Draw import SimilarityMaps
    from scipy.stats import pearsonr
    i=4213
    mol = mols[i]
    v1 = ML_normalized_weights(mol, y_test_preds[i])
    if np.sign(np.sum(v1)) != np.sign(y_test_preds[i]):
        print("ml weights: sign problem detected") 
    V1 = v1*abs(y_test_preds[i]/np.sum(v1))
    V2, v2 = RDkit_normalized_weights(mol)
    print(v1)
    print(V2)
    print("overlap = ", v1@v2, v1@v1, v2@v2)
    print("error = ", y_test_preds[i]-sum(V2))
    print(sum(V2), y_test_preds[i])
    print("pearson's r = ", pearsonr(v1, v2)[0])
    fig1 = get_contour_image(mol, v1)
    fig2 = get_contour_image(mol, v2)

    fig1.savefig("mol5_ML1.pdf", bbox_inches='tight')

    Chem.MolToSmiles(mol)
    s1 = "CC(C)c1noc([C@@H]2CCCN(C(=O)c3ccc(Cl)cc3)C2)n1"
    m1 = Chem.MolFromSmiles(s1)

    Chem.Crippen.MolLogP(m1)

    Chem.Crippen.MolLogP(m2)

    s2 = "CC(C)c1noc([C@@H]2CCCN(C(=O)c3ccccc3)C2)n1"
    m2 = Chem.MolFromSmiles("CC(C)c1noc([C@@H]2CCCN(C(=O)c3ccccc3)C2)n1")
    data2 = create_pytorch_geometric_graph_data_list_from_smiles_and_labels([s1, s2], [3.8662, 3.2128])
    test2 = torch_geometric.loader.DataLoader(dataset = data2, batch_size = 128)
    for (k, batch) in enumerate(test2):
        #print(k)
        # compute current value of loss function via forward pass
        preds = mp.predict(batch, k)
        preds = preds.detach().numpy()

    preds[1]-preds[0]

    s = s1
    mol = Chem.MolFromSmiles(s)
    y = Chem.Crippen.MolLogP(mol)
    x = create_pytorch_geometric_graph_data_list_from_smiles_and_labels([s], [y])
    x = torch_geometric.loader.DataLoader(dataset = x, batch_size = 128)
    for (k, batch) in enumerate(test2):
        #print(k)
        # compute current value of loss function via forward pass
        preds = mp.predict(batch, k)
        preds = preds.detach().numpy()
    y_pred = preds[1]
    v1 = ML_normalized_weights(mol, y_pred)
    if np.sign(np.sum(v1)) != np.sign(y_pred):
        print("ml weights: sign problem detected") 
    V1 = v1*abs(y_pred/np.sum(v1))
    V2, v2 = RDkit_normalized_weights(mol)
    print("overlap = ", v1@v2, v1@v1, v2@v2)
    print("error = ", y_pred-sum(V2))
    print(sum(V2), y_pred)
    print("pearson's r = ", pearsonr(v1, v2)[0])
    fig1 = get_contour_image(mol, V1)
    fig2 = get_contour_image(mol, V2)

    import os
    show_mols([m2], legends=["(5 (no Cl))"], file_name="m5_noCl.pdf")

    Chem.Crippen.MolLogP(m2)-Chem.Crippen.MolLogP(m1)

    preds

    preds

    np.all((data2[1].x == data3[18].x).detach().numpy())

    p = Chem.MolFromSmarts('NC=O')
    show_mols([p])
    print(mol.HasSubstructMatch(p))
    print(mol.GetSubstructMatches(p))



    p = Chem.MolFromSmarts('NC=O')
    amide_present = []
    for i, mol in enumerate(mols):
        if mol.HasSubstructMatch(p):
            amide_present.append(i)
            print(i)

    def delete_multiple_element(list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)

    gcnn_overlaps_no_amide = gcnn_overlaps.copy()
    delete_multiple_element(gcnn_overlaps_no_amide, amide_present)

    plt.hist(gcnn_overlaps, bins=30)

    np.mean(gcnn_overlaps_no_amide)

    print(len(amide_present)/145000)



    print(len(amide_present)/5000)



    plt.scatter(gcnn_overlaps, abs(error_test), s=1)

    from scipy.stats import pearsonr
    pearsonr(gcnn_overlaps, abs(error_test))

    show_mols([train_mols[3]])

    import os
    show_mols(example_mols, legends=["(1)", "(2)", "(3)", "(4)"], file_name="example_mols.pdf")

    np.arange(4)+1

    example_mols = [mols[example_idxs[i]] for i in range(4)]

    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, v1)

    import seaborn as sns
    sns.set()

    np.mean(gcnn_rf_overlaps)

    binwidth=0.1
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(gcnn_overlaps, alpha=0.6, color="blue", bins=np.arange(-1, 1 + binwidth, binwidth), label = "RF/GCNN overlap (average=0.53)")
    ax.legend(fontsize=15, loc='upper left')
    plt.savefig("RF_GCNN_overlap_dist.pdf")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(gcnn_rf_overlaps, gcnn_overlaps, s=1, color="k", label="Pearon's r = 0.30")
    plt.xlabel("overlap (GCNN & RF)", fontsize=14)
    plt.ylabel("overlap (GCNN & ground truth)", fontsize=14)
    ax.legend(fontsize=15, loc='upper left')
    plt.savefig("gcnn-rf_vs_gcnn.pdf")

    from scipy.stats import pearsonr
    print(pearsonr(gcnn_rf_overlaps, test_df.overlap_direct_150000))
    print(pearsonr(gcnn_rf_overlaps, gcnn_overlaps))

    gcnn2_overlaps = pd.read_csv("ML2_overlaps.csv", index_col=0)
    print(gcnn2_overlaps.mean())

    test_df.columns

    np.mean(test_df.overlap_direct_150000_sheri)

    gcnn_dummy_overlap = []
    i=0
    for mol, _y in zip(mols, y_test_preds):
        n_atoms = mol.GetNumAtoms()
        v1 = np.array([_y/n_atoms for _ in range(n_atoms)])
        v2, _, = RDKit_normalized_weights(mol)
        #v2 = ML_normalized_weights(mol, _y)
        #v2, _, _ = RDKit_bit_vector(mol, m_rf)
        v2 = v2/np.linalg.norm(v2)
        v1 = v1/np.linalg.norm(v1)
        overlap1 = v1@v2
        #overlap2 = v1@v3
        gcnn_dummy_overlap.append(overlap1)
        #gcnn_fpa_overlaps.append(overlap2)
        print(i, "overlap = ", overlap1, v2@v2, v1@v1)
        i+=1

    rf_dummy_overlap = []
    i=0
    for mol, _y in zip(mols, test_df.y_pred_150000):
        n_atoms = mol.GetNumAtoms()
        v1 = np.array([_y/n_atoms for _ in range(n_atoms)])
        #v1 = ML_normalized_weights(mol, _y)
        v2, _, = RDKit_normalized_weights(mol)
        v2 = v2/np.linalg.norm(v2)
        v1 = v1/np.linalg.norm(v1)
        overlap1 = v1@v2
        #overlap2 = v1@v3
        rf_dummy_overlap.append(overlap1)
        #gcnn_fpa_overlaps.append(overlap2)
        print(i, "overlap = ", overlap1, v2@v2, v1@v1)
        i+=1

    print(np.mean(gcnn_dummy_overlap))

    fig, ax = plt.subplots(figsize=(9,5))
    binwidth=0.05
    ax.hist(test_df.overlap_direct_150000, alpha=0.6, color="green", bins=np.arange(-0.5, 1 + binwidth, binwidth), label = "Riniker & Landrum (average=0.54)")
    ax.hist(gcnn_overlaps, color='blue', alpha=0.6, bins=np.arange(-0.5, 1 + binwidth, binwidth), label = "GCNN (average=0.47)")
    ax.hist(gcnn_dummy_overlap, color='orange', alpha=0.6, bins=np.arange(-0.5, 1 + binwidth, binwidth), label = "Null-model (average=0.33)")
    #ax.hist(gcnn2_overlaps.ML2_overlaps, color='orange', alpha=0.6, bins=np.arange(-1.0, 1 + binwidth, binwidth), label = "GCNN2 (average=0.45)")
    plt.xlabel("Overlap with ground truth atomic contributions", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    #ax.hist(gcnn_overlaps_no_amide, color='orange', alpha=0.6, bins=np.arange(-1.0, 1 + binwidth, binwidth), label = "GCNN no-amide (average=0.56)")
    plt.legend(fontsize=14.5)
    plt.savefig("Jan_rdkit_GCNN.pdf")

    print(max(gcnn_overlaps), np.argmax(gcnn_overlaps))
    print(max(gcnn2_overlaps.ML2_overlaps), np.argmax(gcnn2_overlaps.ML2_overlaps))
    print(max(test_df.overlap_direct_150000), np.argmax(test_df.overlap_direct_150000))

    !pip install scikit-learn==0.23.2
    import sklearn
    import pickle

    !wget https://sid.erda.dk/share_redirect/hcpheu2AFL
    m_rf = pickle.load(open("/content/hcpheu2AFL", 'rb'))

    smiles = "CC[NH+](CC)[C@@H]1CCN(C(=O)N[C@@H]2CCCC2)C1"
    mol = mols[i]
    ml_weights, atom_weights = get_weights_for_visualization(mol, m_rf, radius=2, n_bits=2048)
    fig = get_contour_image(mol, atom_weights)
    fig = get_contour_image(mol, ml_weights)



    radius = 2
    n_bits = 2048
    bit_overlaps = []
    for i, mol in enumerate(df_test.mols):
        ml_vec, bit_vec = RDKit_bit_vector(mol, radius=radius, n_bits=n_bits)
        ml_vec = ml_vec/np.linalg.norm(ml_vec)
        bit_vec = bit_vec/np.linalg.norm(bit_vec)
        bit_overlap = ml_vec @ bit_vec
        print(i, bit_overlap)
        bit_overlaps.append(bit_overlap)

    def show_mols(mols, mols_per_row = 5, size=200, min_font_size=12, legends=[], file_name=''):
      if legends and len(legends) < len(mols):
        print('legends is too short')
        return None

      mols_per_row = min(len(mols), mols_per_row)  
      rows = math.ceil(len(mols)/mols_per_row)
      d2d = rdMolDraw2D.MolDraw2DSVG(mols_per_row*size,rows*size,size,size)
      d2d.drawOptions().minFontSize = min_font_size
      if legends:
        d2d.DrawMolecules(mols, legends=legends)
      else:
        d2d.DrawMolecules(mols)
      d2d.FinishDrawing()

      if file_name:
        with open('d2d.svg', 'w') as f:
          f.write(d2d.GetDrawingText())
          if 'pdf' in file_name:
            cairosvg.svg2pdf(url='d2d.svg', write_to=file_name)
          else:
            cairosvg.svg2png(url='d2d.svg', write_to=file_name)
          os.remove('d2d.svg')
        
      return SVG(d2d.GetDrawingText())

