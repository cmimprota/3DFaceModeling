# Bonus
- We tried all three repos from the slides.
- There are huge challenges with requirement versions (CUDA, pytorch_geometry etc)
- We decided on pytorch_coma (https://github.com/pixelite1201/pytorch_coma) 
- We modified the repo from Github in the following ways:

## General
- We take obj files as input (our warped meshes) instead of ply

## Training
- We added early stopping
- We modified the network hyperparameters
- We added data augmentation (random translation and mirroring) due to small amount of training data

## Analysis
- We modified the visualisation to fit our face format and rotation
- We added visualisation for the results on the training set
- We compared our results to PCA

## Results
- Train/Test/Val-split: 80/10/10
- Training until epoch 498 with data augmentation, then without 
- Results for epoch 780
- reconstruction error on train set: 0.6182711506291244
- reconstruction error on test set: 1.4994231150073565
- AE: 0.618mm vs PCA: 0.702mm

