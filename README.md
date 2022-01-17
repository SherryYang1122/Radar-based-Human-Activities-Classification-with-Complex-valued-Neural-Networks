# Radar-based Human Activities Classification with Complex-valued Neural Networks
Code used to generate the results for Radar-based Human Activities Classification with Complex-valued Neural Networks paper.
## Input Data Preparation (Matlab)
Matlab codes are for the signal pre-processing pipeline to generate radar datase, including three data formats (range-time, range-Doppler and micro-Doppler spectrograms). NPY files are the input of the neural network for subsequent processing.
Npy-matlab package to save NumPy's NPY format (.npy files) in MATLAB https://github.com/kwikteam/npy-matlab needs to be installed.
## Training and Classification (Python)
Codes are in PyTorch framework to implement a CVNNs package for radar data classification. Plase modify the pathes of three NPY files in the file on your own. 
## Reference
* Trabelsi, Chiheb, et al. "Deep complex networks." arXiv preprint arXiv:1705.09792 (2017)
