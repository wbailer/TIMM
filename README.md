# PyTorch Image Models

This is a fork of [Pytorch Image Models (TIMM)](https://github.com/rwightman/pytorch-image-models), adding more options for training and custom model types. The original TIMM readme with installation instructions can be found [here](README_TIMM.md).

# Modifications of TIMM training and validation

## Training

`--last-layer`: new parameter that freezes the model weights and only trains the classifier parameters

# Validation


`--apr-per-class` calculates accuracy, precision and recall per class

`--acc-pm1` calculates accuracy, counting classes with an index offset of plus/minus 1 as correct

# Dataset preparation scripts

For all datasets a `create_timm_*` script is provided, that creates a folder strucutre from annotions.

## Places365

Place classifcation using the [Places365](http://places2.csail.mit.edu) dataset.

In addition, a script to modify checkpoints to reuse weights in the modified model types is provided.

## NYU Depth v2

Room type classification using the [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset.

Additionally a script to convert the RGB images to JPEG and to create a balanced subset are provided.

## Movienet

Shot type classification using the [Movienet](http://movienet.site/) dataset.

# Additional models

Variants of EfficentNet-B3 for specific training configurations on Places365 have been added:

`efficientnet_b3_places365supercat`: Add a layer to predict the Places365 supercategries from a weighted sum of the probabilties of the more fine-grained classes.

`efficientnet_b3_places365supercatmax`: Add a layer to predict the Places365 supercategries from fine-grained class with the highest probability.



# Acknowledgement

<img src="img/Tailored_Media_Logo_Final.png" width="200">

The research leading to these results has been funded partially by the program ICT of the Future by the Austrian Federal Ministry of Climate Action, Environment, Energy, Mobility, Innovation and Technology (BMK) in the project [TailoredMedia](https://www.joanneum.at/en/digital/reference-projects/tailoredmedia). 

<img src="img/BMK_Logo_srgb.png" width="200"><img src="img/FFG_Logo_DE_RGB_1000px.png" width="200">