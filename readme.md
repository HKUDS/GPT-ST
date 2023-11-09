# GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

A pytorch implementation for the paper: [GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks](https://arxiv.org/abs/2311.04245v1)<br />  

Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang* (*Correspondence)<br />  

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, [South China University of Technology](https://www.scut.edu.cn/en/), PAZHOU LAB  

This repository hosts the code, data, and model weights of **GPT-ST**. Furthermore, it also includes the code for the baselines used in the paper.

## Introduction

<p style="text-align: justify">
GPT-ST is a generative pre-training framework for improving the spatio-temporal prediction performance of downstream models. The framework is built upon two key designs: (i) We propose a spatio-temporal mask autoencoder as a pre-training model for learning spatio-temporal dependencies. The model incorporates customized parameter learners and hierarchical spatial pattern encoding networks, which specifically designed to capture spatio-temporal customized representations and intra- and inter-cluster region semantic relationships. (ii) We introduce an adaptive mask strategy as part of the pre-training mechanism. This strategy guides the mask autoencoder in learning robust spatio-temporal representations and facilitates the modeling of different relationships, ranging from intra-cluster to inter-cluster, in an easy-to-hard training manner.  
</p>

![The detailed framework of the proposed GPT-ST.](https://github.com/LZH-YS1998/GPT-ST_img/blob/main/fig3.png)


## Code structure
* **conf**: This folder includes parameter settings for GPT-ST (`GPTST_pretrain`) as well as all other baseline models.
* **data**: The documentation encompasses all the datasets utilized in our work, alongside prefabricated files and the corresponding file generation codes necessary for certain baselines.
* **lib**: Including a series of initialization methods for data processing, as follows:
	* `Params_xxx.py`: To configure the parameters for GPT-ST and the baseline models。
	* `TrainInits.py`: Training initialization, including settings of optimizer, device, random seed, etc.
	* `add_window.py`: Time series slicing。
	* `dataloader.py` and `load_dataset.py`: Load, split, generate data, etc.
	* `logger.py`: For output printing。
	* `metrics.py`: Method for calculating evaluation indicators。
	* `normalization.py`: Normalizationmethod。
	* `predifineGraph.py`: Predefined graph generation method。
* **model**: Includes the implementation of GPT-ST and all baseline models, along with the necessary code to support the framework's execution. The `args.py` script is utilized to generate the required prefabricated data and parameter configurations for different baselines. Additionally, the `SAVE` folder serves as the storage location for saving the pre-trained models.


## Environment requirement
The code can be run in the following environments, other version of required packages may also work.
* python==3.9.12
* numpy==1.23.1
* pytorch==1.9.0
* cudatoolkit==11.1.1  

Or you can install the required environment, which can be done by running the following commands:
```
# cteate new environmrnt
conda create -n GPT-ST python=3.9.12

# activate environmrnt
conda activate GPT-ST

# Torch with CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install required libraries
pip install -r requirements.txt
```


## Run the codes 
* First, enter "data" folder to unzip all zip files, and then enter "model" folder:
```
cd model
```
* To test different models in various modes, you can execute the Run.py code. There are some examples:
```
# Evaluate the performance of STGCN enhanced by GPT-ST on the PEMS08 dataset
python Run.py -dataset PEMS08 -mode eval -model STGCN

# Evaluate the performance of ASTGCN enhanced by GPT-ST on the METR_LA dataset
python Run.py -dataset METR_LA -mode eval -model ASTGCN

# Evaluate the original performance of CCRNN on the NYC_TAXI dataset
python Run.py -dataset NYC_TAXI -mode ori -model CCRNN

# Pretrain from scratch on NYC_BIKE dataset, checkpoint will be saved in model/SAVE/NYC_BIKE/new_pretrain_model.pth
python Run.py -dataset NYC_BIKE -mode pretrain
```

* Parameter setting instructions. The parameter settings consist of two parts: the pre-training model and the baseline model. To avoid any confusion arising from potential overlapping parameter names, we employ a hyphen (-) to specify the parameters of GPT-ST and use a double hyphen (--) to specify the parameters of the baseline model. Here is an example:
```
# Set first_layer_embedding_size and out_layer_dim to 32 in STFGNN
python Run.py -model STFGNN -mode eval -dataset PEMS08 --first_layer_embedding_size 32 --out_layer_dim 32
```



## Citation
```
@inproceedings{
li2023generative,
title={Generative Pre-Training of Spatio-Temporal Graph Neural Networks},
author={Zhonghang Li and Lianghao Xia and Yong Xu and Chao Huang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=nMH5cUaSj8}
}
```

## Acknowledgements
We developed our code framework drawing inspiration from [AGCRN](https://github.com/LeiBAI/AGCRN) and [STEP](https://github.com/zezhishao/STEP). Furthermore, the implementation of the baselines primarily relies on a combination of the code released by the original author and the code from [LibCity](https://github.com/LibCity/Bigscity-LibCity). We extend our heartfelt gratitude for their remarkable contribution.
