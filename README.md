# LA-HDR:
**Light Adaptive HDR Reconstruction Framework for Single LDR Image Considering Varied Light Conditions**

_Xiangyu Hu, Liquan Shen, Mingxing Jiang, Ran Ma and Ping An_
_School of Communication and Information Engineering, Shanghai University_

[![DOI](https://zenodo.org/badge/DOI/10.1109/TMM.2022.3183404.svg)](https://doi.org/10.1109/TMM.2022.3183404)
This is the official repository for LA-HDR: Light Adaptive HDR Reconstruction Framework.  

We provide two types of reference implementations for LA-HDR: TensorFLow & PyTorch. Please check the respective repos for details.

Meanwhile, considering the ambiguities in evaluating HDR-VDP2, we provide our HDR-VDP2 evaluation implementation for reference. See `./hdrvdp-2.2.1/metric_vdp2.m` for details.

## Notes

1. LA-HDR is strongly suggested to be trained seperately by each sub-networks sequentially. 

2. As the whole solution is focused on HDR reconstruction, we do not put much efforts on low-light denoising, thus the denoising effect of _DeNet_ is rather poor. The pre-trained model for _DeNet_ is not provided in PyTorch version. Users may evaluate its performance in TensorFlow one. 

3. The training parameters in each training script are provided only as the reference setup.

4. Users are suggested to develop a better denoising solution to replace _DeNet_, or maybe just replace it with solutions like _CycleISP_, etc. Recent low-light denoising solutions tend to consider the denoising problem in the high bit-depth / RAW domain, therefore, _EnhanceNet_ is designed in LA-HDR for helping such denoising measures. The `hist_block` module in _EnhanceNet_ can be considered to deal with various ISP strategies. Such improvement are not investigated in this LA-HDR version.

## Citing

**Plain Text:**

X. Hu, L. Shen, M. Jiang, R. Ma and P. An, "LA-HDR: Light Adaptive HDR Reconstruction Framework for Single LDR Image Considering Varied Light Conditions," _IEEE Transactions on Multimedia_, 2022.

**BibTex:**
```
@ARTICLE{lahdr_2022,
  author={Hu, Xiangyu and Shen, Liquan and Jiang, Mingxing and Ma, Ran and An, Ping},
  journal={IEEE Transactions on Multimedia}, 
  title={LA-HDR: Light Adaptive HDR Reconstruction Framework for Single LDR Image Considering Varied Light Conditions}, 
  year={2022},
  pages={1-16},
  doi={10.1109/TMM.2022.3183404}}
```
