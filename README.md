# Radar-based Human Activities Classification with Complex-valued Neural Networks
Code used to generate the results for the paper ‘Radar-based Human Activities Classification with Complex-valued Neural Networks’. This has been accepted and will be presented at the IEEE Radar Conference 2022.

This work was performed as part of an MSc thesis project at TU Delft; please see https://repository.tudelft.nl/islandora/object/uuid%3A5cfd6c16-4db6-45c7-88d6-34acf44f8848?collection=education. Disclaimer: the scripts have originated from a student project and have not been optimised for professional usage.  

## Raw Data Download
The data processed for this work are available at https://data.4tu.nl/articles/dataset/Dataset_of_continuous_human_activities_performed_in_arbitrary_directions_collected_with_a_distributed_radar_network_of_five_nodes/16691500/2


## Input Data Preparation (Matlab)
Matlab scripts are for the signal pre-processing pipeline to generate the radar dataset, including 3 data formats (range-time, range-Doppler and micro-Doppler spectrograms). NPY files are the input of the neural network for subsequent processing. 
Note that the Npy-matlab package to save NumPy's NPY format (.npy files) in MATLAB https://github.com/kwikteam/npy-matlab needs to be installed.

## Training and Classification (Python) 
Scripts are written in PyTorch framework to implement a CVNNs package for radar data classification. 
Install requirements for computer vision experiments with pip:
```
!pip install torch==1.7.1
pip install complexPyTorch
```
Run models:
```
!python train.py --path '.../rt_complex2.npy'
```
Please modify the paths of the three NPY files (range-time, range-Doppler and micro-Doppler spectrograms) in the file on your own to get the results. Other arguments may be added as well, such as batchsize and epochs.

## Reference
* Trabelsi, Chiheb, et al. "Deep complex networks." arXiv preprint arXiv:1705.09792 (2017)
* Guendel, Ronny Gerhard; Unterhorst, Matteo; Fioranelli, Francesco; Yarovoy, Alexander (2021): Dataset of continuous human activities performed in arbitrary directions collected with a distributed radar network of five nodes. 4TU.ResearchData. Dataset. https://doi.org/10.4121/16691500.v2 
