# Do machines dream of atoms?


<a href="https://colab.research.google.com/drive/1dVsmJlGqaHykemUCLh--lfoppsbNv_I9?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The data and code made available here is described in:

* Maria H. Rasmussen, Diana S. Christensen and Jan H. Jensen. "Do machines dream of atoms? A quantitative molecular benchmark
for explainable AI heatmaps", *ChemRxiv* (Mar. 2022)

## Models

The 9 different RF models described in the above mentioned paper is available here: [RF_models](https://sid.erda.dk/sharelink/eUVFpTDU62)

Each model has 200 trees, with a minimum sample per leaf node of 3 and ECFP4 with a bit vector length of 2048 as the input features.

They are named "Ntrain_200_3_2048.p" where Ntrain is any of the 9 training set sizes used to train the model: 100, 500, 1000, 5.000, 10.000, 20.000, 50.000, 100.000 or 150.000 

The training data for each model is the first Ntrain entries of ZINC_250k.smi

## Test data

test_data.csv contains smiles, RDKit mol objects and Crippen logP values for the test set of 5000 molecules. 

It also contains predictions, atomic overlaps and FPA overlaps for the 9 different RF models. 

For the predictions, the columns are named: y_pred_Ntrain, where Ntrain is the size of the training data, the model was trained on (100, 500, 1000, 5.000, 10.000, 20.000, 50.000, 100.000 or 150.000). 

Likewise, the overlap between the ML atomic attribution vector and the atomic attribution from the Crippen logP model is given for the 9 different models, the columns are named: overlap_atom_Ntrain. 

Lastly, the overlap between the ML atomic attribution vector and the fingerprint adjusted (FPA) attribution vector from the Crippe logP value is given for each model, named as: overlap_fpa_Ntrain 

The test data is the last 5000 entries of ZINC_250k.smi

## Visualizing atomic contributions

example_notebook.ipynb contains examples of how to create contour plots like those found in the paper above
