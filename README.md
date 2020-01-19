# VUB - Data Representation, Reduction and Analysis Project
------
# Image Super Resolution - Using ESPCN

<!-- TOC -->

  - [Requirements](#requirements)
    - [For training](#for-training)
    - [For Upscaling](#for-upscaling)
  - [How to Train](#how-to-train)
  - [Upscaling Images](#upscaling-images)
    - [Super_res.py commands](#super_respy-commands)
    - [Batch_SR-PSNR_calculation.py](#batch_sr-psnr_calculationpy)
    - [Batch_bicubic_PSNR_calculation.py](#batch_bicubic_psnr_calculationpy)

<!-- /TOC -->

## Requirements
### For training
At least one image dataset contained within a zip file

### For Upscaling
At least one image and a model

## How to Train 
```sh
python train.py --upscale 3 --trainBatchSize 4 --validBatchSize 100 
--nEpochs 30 --lr 0.001 --cuda --cropSize 420 --func leaky
```

Passing this code to the terminal/command line/powershell will start the process.  
After that the code will ask for:
- Training dataset location (including the zip file)  
It accepts relative paths or direct paths (ex: `dataset\training_imageset.zip`)
- Whether the images are contained within a folder or not (1/0)
  - If yes: ask for folder structure until images are found  
    (ex: `training_folder\another_folder`)
- Will repeat process above for validation datasets.

The results will be saved in a folder of `logs_scale_#UPSCALE#_crop_#CROPSIZE#`
 
**IMPORTANT:** 
- For next training, rename or move the result folder.    
(Else, the results will be overwritten)
- If the code fails, remove the result folder and extr folder from root directory.  
(existence of extr folder will yield an error, stopping the code)

## Upscaling Images
There are two files that can be used to upscale images:  
**super_res.py** and **batch_SR-PSNR_calculation.py**

### Super_res.py commands
```sh
python super_res.py --input LowRes-Image.png 
--model processed\logs-models_scale_%i\epoch_%i_model.pth 
--output SuperRes-Image.png --cuda
```

The arguments are straightforward, pass the path of input image, model and where the output is desired.

There is an optional parameter `--compare HighRes-Image.png` where it will print out the PSNR of upscaled image compared to the given High Resolution Image.

### Batch_SR-PSNR_calculation.py
```sh
python batch_SR-PSNR_calculation.py
--model processed\logs-models_scale_%i\epoch_%i_model.pth
--cuda
```

Unlike super_res.py file, it only takes the model as argument.
- Then it will first ask for image set to be upscaled
- And high resolution counterparts of the low res images.

Resulting SR images are saved in `SR_results` folder with the PSNR result logs.

### Batch_bicubic_PSNR_calculation.py
The way this works is identical to batch_SR-PSNR_calculation.py  
Only difference is that while running in terminal it only expects

`--cuda` as an argument, no models required.  
It will ask for low-res image sets and high-res image sets so that it will upscale the low-res images by using BICUBIC and calculate the PSNR.

Note that this file will not output the BICUBIC upscaled images, only the PSNR calculations.

## Directory Description

| Directory Name   | Contents                                                                         |
| ---------------- | -------------------------------------------------------------------------------- |
| .best_processed  | Best result yielding models parameters (cropSize 420, trained on X3 bicubic set) |
| .best_processed  | Files that are trained with various parameters and logs                          |
|                  | The parameters are logged in 'Folder Index ReadMe.txt'                           |
|                  | Also contains the PSNR logs for BICUBIC Upscaling                                |
| compare-contrast | 3 images to calculate PSNR results wrt our trained models                        |
|                  | PSNR results are found in SR_results.txt                                         |
| scripts          | small python files created to not convolute any main Files                       | 
