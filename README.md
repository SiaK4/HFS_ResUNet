# Mitigating Spectral Bias in Neural Operators via High-Frequency Scaling for Physical Systems

This repository contains the code for our paper by Siavash Khodakarami, Vivek Oommen, Aniruddha Bora, and George Em Karniadakis.

**"Mitigating Spectral Bias in Neural Operators via High-Frequency Scaling for Physical Systems"**  
[*Read the paper here:*] (https://arxiv.org/abs/2503.13695)  
The code is implemented using **PyTorch**.

---

## Requirements

The codes are tested and compatible with Python 3.9.16 and the following dependencies:  
Pytorch 2.4.1 + Cuda 12.1 toolkit  
Numpy 1.24.1  

---

## Training
First, download the data and put them in the correct directory (see dataset section)
Run train_HFS.py for training the subcooled pool boiling problem.  
Run train_kolmogorov_HFS.py for training the Kolmogorov flow problem.

---
## Dataset
Prior the running the training scripts, download the data and place them in the following directories:  
a) Subcooled pool boiling data: './BubbleML/PoolBoiling-SubCooled-FC72-2D'  
b) Kolmogorov flow data: './kolmogorov_data.npz'  

##### Download the Subcooled Pool Boiling study consisting of 10 simulations (source: https://github.com/HPCForge/BubbleML/tree/main)  
wget https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/pool-boiling-subcooled-fc72-2d.tar.gz  
tar -xvzf pool-boiling-subcooled-fc72-2d.tar.gz && rm pool-boiling-subcooled-fc72-2d.tar.gz  

##### Download the Kolmogorov flow problem (data to be shared soon...)  
Follow the notebook from this source: https://github.com/vivekoommen/NeuralOperator_DiffusionModel/blob/main/case_1_kolmogorov/data/ns_data_generator.ipynb  

---
# Checkpoints
Checkpoints for subcooled pool boiling models with and without high-frequency scaling are included in "Ckpts" directory.  
Checkpoints for Kolmogorov flow problem will be shared soon.  
Checkpoints for saturated pool boiling problem will be shared soon.

---
# Remark  
Based on our experience, the subcooled pool boiling problem with the problem setup described in the paper converges in less than 1000 epochs (for a model with 3.5 M parameter).  
Based on our experience, the Kolmogorov flow problem with the problem setup described in the paper converges in less than 300 epochs (for a model with 1.7 M parameter).

