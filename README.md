# VUB-DRRA-Project2019
Data Representation, Reduction and Analysis Project: Image Super Resolution - Using ESPCN 

**Example of how to train:**
`python train.py --upscale 3 --trainBatchSize 4 --validBatchSize 100 --nEpochs 30 --lr 0.001 --cuda`

Then input relative dataset location:  
dataset\training_imageset.zip

If zip includes folders input 1, else 0.

Then it will ask for folder structure until you reach the images:  
training_folder1\folder2

And repeat for validation datasets

**Example of how to Super Resolution:**
`python super_res.py --input LowRes-Image.png --model processed\logs-models_scale_X\epoch_XX_model.pth --output SuperRes-Image.png --cuda`

X = upscale from previous input
XX = number of epoch model that is preferred to use in SR

