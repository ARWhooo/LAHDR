# LA-HDR:
**Light Adaptive HDR Reconstruction Framework for Single LDR Image Considering Varied Light Conditions**

_Xiangyu Hu, Liquan Shen, Mingxing Jiang, Ran Ma and Ping An_
_School of Communication and Information Engineering, Shanghai University_

[![DOI](https://zenodo.org/badge/DOI/10.1109/TMM.2022.3183404.svg)](https://doi.org/10.1109/TMM.2022.3183404)
This is the PyTorch version of the reference implementation for LA-HDR: Light Adaptive HDR Reconstruction Framework.  


## Prerequisites

```
torch==1.71
torchvision==0.8.2
torchstat
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
transplant (Optional, if you want to disable VDP2 evaluation. Modify ./utils/metrics.py to remove this dependency.)
```

## Evaluation

Run `main.py --stage test --config recipe_LAHDR` in Python IDE (e.g., Spyder) or command line. You may modify the exact configure entries in the evaluation script `recipe_LAHDR.py` beforehand, which are listed in dict `configs -> dataset -> test`: 

1. dataset_type: the exact `dataset` class in the ./dataset package designated for loading the LDR and HDR pairs;

2. data_source: the H5 file that contains the LDR and HDR pairs;

3. fetch_keys: H5 key entries that stores the images (H5 entry `/HDR `stores the HDR images, and `/LDR` stores the corresponding LDR images, hence the list `HDR, LDR`);

4. argumentations: the argumentation configs for the input images. See `./dataset/argumentation.py` for detailed process.

`Note: `Different dataset classes may require different entries for this dict. See `./dataset/__init__py` for details. 

## Training

The whole LA-HDR framework can be trained sequentially as follows: 

1. Train _EnhanceNet_ : execute `main.py --stage train --config recipe_EnhanceNet`. 

2. Train _EBNet_ : execute `main.py --stage train --config recipe_EBNet`. 

3. Train _DeNet_ : execute `main.py --stage train --config recipe_DeNet`(Optional). 

4. Train _FuseNet_ : execute `main.py --stage train --config recipe_FuseNet`. 

Make sure that the common training parameters in each of these scripts like "model_name", "dataset", "epochs", "batch_size", "learning_rate", etc. are properly configured before training. Users may check the comments in `recipe_EnhanceNet.py` for help.

## Notes

1. LA-HDR is strongly suggested to be trained seperately by each sub-networks sequentially. 

2. As the whole solution is focused on HDR reconstruction, we do not put much efforts on low-light denoising, thus the denoising effect of _DeNet_ is rather poor. The pre-trained model for _DeNet_ is not provided in the PyTorch version. Users may evaluate its performance in TensorFlow one. Users are suggested to develop a better denoising solution to replace _DeNet_, or maybe just replace it with solutions like _CycleISP_, etc. 

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
