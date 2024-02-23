# TRCN & ATFRSD

PyTorch implement of [Boiler Furnace Temperature Field Measurement and Reconstruction Error Elimination Based on Temperature Field Residual Correction Network](https://ieeexplore.ieee.org/document/10399821)

__TRCN__: Temperature Field Residual Correction Network

__ATFRSD(-W)__: Acoustic Temperature Field Reconstruction Simulation Dataset (with Water-cooled Walls)

_*TRCN is described in section II-B&C and shown in Fig.3-5 of the paper._<br/>
_**ATFRSD(-W) is described in section III-A._

****

## Experiments

### Environment

> Ubuntu 18.04.5 LTS
>> python = 3.8.0 <br/>
> PyTorch = 1.11.0 <br/>
> numpy = 1.22.3

### Dataset

_ATFRSD_ and _ATFRSD-W_ are located in the folder data and are named separately. The temperature fields in _T.mat_ or _TT.mat_ is to be measured, is also called target fields. The temperature fields in _TR.mat_ or _TTR.mat_ is reconstructed by traditional method, and its details are described in the paper. 

_*You can also generate datasets tailored to your needs based on the details provided in our paper, as long as abiding by academic norms._

### Parameters

You can use the default parameters run. The following configurations have been defined as default parameters, and you do not need to configure or change them. 

> batchSize = 32 <br/>
> epochs = 50 <br/>
> milestone = 30 <br/>
> lr = 1e-3

_num_of_layers_ is the total number of TRCN layers, and the default setting in the code is the same as in the paper, which is 17.
> num_of_layers = 17


### Results

All results involving TRCN and ATFRSD(-W) have been disclosed in the paper.

****

## Cite

If this work has provided you with a reference, please cite our article.
>@ARTICLE{10399821,
  author={Duan, Yixin and Chen, Liwei and Zhou, Xinzhi and Shi, Youan and Wu, Nan},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Boiler Furnace Temperature Field Measurement and Reconstruction Error Elimination Based on Temperature Field Residual Correction Network}, 
  year={2024},
  volume={73},
  number={},
  pages={1-15},
  keywords={Temperature measurement;Acoustic measurements;Pollution measurement;Measurement uncertainty;Boilers;Temperature distribution;Reconstruction algorithms;Acoustic temperature measurement;convolutional neural network;error elimination;ill-conditioned problem;temperature field construction},
  doi={10.1109/TIM.2024.3353873}}

****

## Others

If you have any questions about TRCN and ATFRSD, please contact the [authors](https://ieeexplore.ieee.org/document/10399821/authors) by email.


