# mRS prediction
The data and code for the paper "Vision-Language Model with Siamese Bilateral Difference Network for Acute Ischemic Stroke Outcome Prediction" submitted to CVPR2025. <br />

## Requirements
imbalanced_learn 0.12.0<br />
imblearn 0.0<br />
monai 1.3.0<br />
numpy 2.1.3<br />
pandas 2.2.3<br />
scikit_learn 1.4.1.post1<br />
SimpleITK 2.3.1<br />
torch 2.2.1<br />
transformers 4.44.2<br />

## Getting Started

### Installation
Create and activate a new conda environment
```
conda create -n mrs python=3.9.18
conda activate collateral
```

Install Dependencies
```
pip install -r requirements.txt
```

## mRS Prediction on CT Angiography
### Skull-stripping
  
After converting the DICOM files to NIfTI format, perform skull stripping according to the instructions at https://github.com/WuChanada/StripSkullCT.  <br />

### Training  
```
python train.py --save-path <Path> --num-classes <Classes>
```
Parameter Description：  
* --save-path <Path>: Model save path, please specify a valid directory
* --num-classes <Classes>: Classification category, please specify a positive integer, such as 3

### Testing  
```
python test.py --checkpoint-path <Path> --checkpoint-name <list> --num-classes <Classes> --k-fold <fold>
```
Parameter Description：  
* --checkpoint-path <Path>: Directory for saving model checkpoints
* --checkpoint-name <list>: Name of each folded test model, such as 5-fold cross-validation: ['fold0.pt', 'fold1.pt', 'fold2.pt', 'fold3.pt', 'fold4.pt']
* --num-classes <Classes>: Classification category, please specify a positive integer, such as 3

### Reproduction details and codes
During reproduction, for the CNN-based methods, Transformer-based methods, and Hybrid-CNN-Transformer-based methods. All of these methods can be found at [[Baseline]](./baseline).  <br />
Note that for all compared methods, to perform fair comparisons, we used the same data split and five-fold cross-validation.  <br />



