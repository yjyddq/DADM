# DADM
The official implementation of [**"DADM: Dual Alignment of Domain and Modality for Face Anti-spoofing (DADM)"**](https://arxiv.org/abs/2503.00429).

:
<div align=center>
<img src="https://github.com/yjyddq/DADM/blob/main/assets/Motivation.png" width="512" height="224" />
</div>

An overview of the proposed DADM architecture:

<div align=center>
<img src="https://github.com/yjyddq/DADM/blob/main/assets/Architecture.png" width="892" height="384" />
</div>

## Congifuration Environment
- python 3.10 
- torch 1.12.1 
- torchvision 0.13.1
- cuda 11.4

## Data

**Dataset.** 

Download the WMCA, CASIA-SURF, CASIA-CeFA, and PADISI-USC datasets.

**Data Pre-processing.** 

Please refer to [Data Preprocess](https://github.com/yjyddq/DADM/blob/main/data_preprocess). 


## Training and Testing

Run like this:
```python
CUDA_VISIBLE_DEVICES=0 
python train_cross_lx.py --lr 5e-5 --batchsize 16 --modality RGBDIR --model dadm --train SURF CeFA USC --test WMCA
```

## Citation
Please cite our paper if the code is helpful to your research.
```
@article{yang2025dadm,
  title={Dadm: Dual alignment of domain and modality for face anti-spoofing},
  author={Yang, Jingyi and Lin, Xun and Yu, Zitong and Zhang, Liepiao and Liu, Xin and Li, Hui and Yuan, Xiaochen and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2503.00429},
  year={2025}
}
```