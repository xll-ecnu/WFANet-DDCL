
official code of paper "WFANet-DDCL: wavelet-based frequency attention network and dual domain consistency learning for 7T MRI synthesis from 3T MRI" in TCSVT
## Wavelet-based frequency attention net and Dual domain consistency learning
Exploring wavelet-based frequency attention encoding network for Semi-Supervised 7-Tesla MR images synthesis from 3-Tesla MR images


## Requirements
* [Pytorch]
* TensorBoardX
* Efficientnet-Pytorch
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......


## Usage

1. Clone the repo:
```
git clone https://github.com/xll-ecnu/WFANet-DDCL.git
```
2. Prepare the processed 3T and 7T MRI data and put the data in `../data/paired10`.
  
data/
    ├── paired10
    │     └── images
    │     └── train.txt
    │     ├── val.txt
    │     ├── test.txt (optional)


3. Train the model
```
cd code
python train.py --root_path ../data/XXX --exp paired10/XXX --model WFANet -max_iterations XXX -batch_size XXX --base_lr XXX --num_classes 1 --paired_num XXX
```

4. Test the model
```
python val.py -root_path ../data/XXX --exp paired10/XXX -model WFANet --num_classes 1 --paired_num XXX
```

You can choose model, dataset, experiment name, iteration number, batch size and etc in your command line, or leave it with default option.

