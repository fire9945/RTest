# Attentive Pooling-Based Weighted Sum of Spectral Decay Rates for Blind Estimation of Reverberation Time 


![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=yellow)
![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=black)


![header](https://capsule-render.vercel.app/api?type=Rect&color=auto&height=200&section=footer&text=Blind%20T60%20Estimation&fontSize=70&textBg=true)


## About
This repo contains the implementation and experimetal results of the Attentive pooling-based Weighted Sum of Spectral Decay Rates (AWSSDR). The AWSSDR is a environmental feature for blind estimation of reverberation time.


## Training
### python CUBLAS_WORKSPACE_CONFIG=:4096:8 1_trainset*.py

#### The trained model is stored in checkpoints/'model_name.dnn'

## Evaluation
### python 1_testset*.py

