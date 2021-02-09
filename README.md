# Valid, Sparse, and Stable Explanations in Graph Neural Networks

##### by Anonymous Authors


## 1. Requirements

See `requirements.txt` for the main python packages used to run this repository with `python 3.7`.

## 2. Data

The real-world datasets will be downloaded via `pytorch-geometric`. 
For the synthetic dataset, we included the file `generate_gnnexplainer_dataset.py` and in `data/syn2.npz` our resulting graph. 

## 3. Execute Zorro

You can simply run
```
python3 execution.py
```
to get explanations for the default setting: 10 nodes for Cora and GCN with tau=0.85.

We included the save points of the model and the randomly selected nodes in the `results` directory. 

## 4. Evaluation

Running `evaluate_zorro.py` will create csv files with the evaluate explanations. 