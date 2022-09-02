# LA-HDR:
**Light Adaptive HDR Reconstruction Framework for Single LDR Image Considering Varied Light Conditions**

_Xiangyu Hu, Liquan Shen, Mingxing Jiang, Ran Ma and Ping An_
_School of Communication and Information Engineering, Shanghai University_

[![DOI](https://zenodo.org/badge/DOI/10.1109/TMM.2022.3183404.svg)](https://doi.org/10.1109/TMM.2022.3183404)
This is the Tensorflow version of the reference implementation for LA-HDR: Light Adaptive HDR Reconstruction Framework.  


## Prerequisites

```
tensorflow==1.10
numpy
scikit-image
opencv-python
openexr
scipy
matplotlib
exifread
rawpy
logging
h5py 
transplant (Optional, only required if you want to enable VDP2 evaluation.)
```

## Evaluation

Just run `LAHDR_test.py` in Python IDE (e.g., Spyder) or command line. You may modify the preset parameters in this evaluation script beforehand: 

```
denoise : set 'True' to invoke extra denoising step by DeNet ;
eb_shift : set 'True' to adaptively offset the exposure of the input image by using EBNet ;
inp_dir : directory of the input LDR samples;
outlog : log of the exposure offset value predicted from EBNet ;
out_dir : output directory of the predicted HDR samples.
```

## Training

The whole LA-HDR framework can be trained sequentially as follows: 

1. Train _EnhanceNet_ : using the script `enhance_train.py`. 

2. Train _EBNet_ : using the script `eb_train.py`. 

3. Train _DeNet_ (Optional) : using the script `de_train.py`. 

4. Train _FuseNet_ : using the script `fuse_train.py`. 

Make sure that the common training parameters in each of these scripts like "train_dir", "test_dir", "epochs", "batchsize", "learning_rate", etc. are properly configured before running these scripts.

```
train_dir : the training sample and label pairs (contained in a single H5 file, loaded by SingleDataFromH5 or SelectedKVLoader in dataloader.py);
test_dir : the evaluation sample and label pairs.
```

## Notes

1. LA-HDR is strongly suggested to be trained seperately by each sub-networks sequentially. 

2. As the whole solution is focused on HDR reconstruction, we do not put much efforts on low-light denoising, thus the denoising effect of _DeNet_ is rather poor.  Users are suggested to develop a better denoising solution to replace _DeNet_, or maybe just replace it with solutions like _CycleISP_, etc. 

3. The training parameters in each training script are provided only as the reference setup.

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
