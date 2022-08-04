# UGPCL

> [IJCAI' 22] Uncertainty-Guided Pixel Contrastive Learning for Semi-Supervised Medical Image Segmentation.
> 
> Tao Wang, Jianglin Lu, Zhihui Lai, Jiajun Wen and Heng Kong

![](pics/overview.jpg)

# Installation
Please refer to [requirements.txt](requirements.txt)

# Dataset
*ACDC* dataset can be found in this [Link](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

Please change the dataset root directory in _configs/\_datasets/acdc_224.yaml_.

# Results

### *ACDC* (136 labels | 10%):

|    Method    |  Model   | Iterations | Batch Size | Label Size |  DsC  |                             Ckpt                             |                     Config File                     |
| :----------: | :------: | :--------: | :--------: | :--------: | :---: | :----------------------------------------------------------: | :-------------------------------------------------: |
| UGPCL | UNet-R50 |    6000    |     16     |     8      | 88.11 | [Link](https://drive.google.com/file/d/1T8T6g_xiJWGetQhZeFMNG2q7dzmYyN4s/view?usp=sharing) |   configs/comparison_acdc_224_136/ugpcl_unet_r50.yaml  |
|    Mean Teacher     | UNet-R50 |    6000    |     16     |     8      | 85.75 | [Link](https://drive.google.com/file/d/1mWKKoeZbSlf6DNxqnoypr50ialPMqFYL/view?usp=sharing) | configs/comparison_acdc_224_136/mt_unet_r50.yaml |

### Visualization

- Segmentation results:

  <img src="pics/preds.jpg" style="zoom: 15%;" />

- Pixel features (t-SNE):

  <img src="pics/show_feats.jpg" style="zoom: 20%;" />


# Reference
- [https://github.com/HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS)
- [https://github.com/tfzhou/ContrastiveSeg](https://github.com/tfzhou/ContrastiveSeg)

# Citation
```bibtex
@inproceedings{ijcai2022-201,
  title     = {Uncertainty-Guided Pixel Contrastive Learning for Semi-Supervised Medical Image Segmentation},
  author    = {Wang, Tao and Lu, Jianglin and Lai, Zhihui and Wen, Jiajun and Kong, Heng},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {1444--1450},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/201},
  url       = {https://doi.org/10.24963/ijcai.2022/201},
}
```
