# CCT-ColorSpace
Repository for the project I did for the Deep Learning course. The project is titled "Effect of Image Color Space on the robustness of Compact Convolutional Transformers".

## Other repositories
The main experimentation pipeline was adapted from https://github.com/nis-research/afa-augment

The CCT implementation was provided by https://github.com/SHI-Labs/Compact-Transformers (original implementation)

## Own addition to the work
Given that I have worked on this course as part of my PhD, I have changed the pipeline quite a lot for the course and other research in parallel. I have indicated my additions for this course in particular as follows in every file:

```
################################# Added for DL #########################################

code...

#################################  Until here  #########################################
```

For the src.utils folder, only changes have been made to transformers.py

## Reproducibility:
The seed for all experiments was set to 88. To run the experiments, one needs to change the api key in the main.py and evaluate.py file. Evaluation requires a path to the checkpoint of the model to passed in the bash file.

WandB plots available upon request.
