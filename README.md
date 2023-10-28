# PC-DNN-4Turbulent-Flows
Physics Constrained Deep Neural Network for predicting Mean Turbulent Flows.


## Project Purpose
CFD simulations are vital in various fields but can be computationally expensive due to complex equations. To address this, I propose PC-DNN, a physics informed CNN-based model that efficiently approximates solutions for non-uniform steady laminar flows. It learns complete solutions of the Navier-Stokes equations from CFD data, providing a speedup of up to orders of magnitude with low error rates.

## Problem Geometry

![Part 1](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/scheme.png)

## Architecture

![Part 2](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/U-Net.PNG)

Image Input is of shape (128, 256, 1). The above classical U-Net Architecture is employed in the problem. One Main difference is that we are using three different decoders from the latent space. **So, we have one encoder and 3 separate decoders!**

## Loss Function
The Loss Function is taken as the mean squared of three error terms - 1.Two momentum equations (in x & y directions) 2. Continuity loss

**Continuity Equation -**

![Part 3](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/continuity%20equation.png)

**Momentum Equations -**

![Part 4](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/momentum%20equation.png)

**Weighted Custom Loss -** frac*mse_loss + (1-frac)*nv_loss, mse_loss = pure data driven loss, nv_loss = cont_loss_squared + momentum_loss_squared

## Model Architecture

![Part 5](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/architecture.png)

## Results

![Part 6](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/some_results.png)

## Model-Deployement(HuggingFace Application)

![Part 7](https://github.com/Khalid-Rafiq-01/PC-DNN-4Turbulent-Flows/blob/main/Images/hugginface_model.gif)

Visit the app on [HuggingFace](https://huggingface.co/spaces/krafiq/deep-neural-networks-for-navier-stokes-equations) to draw any shape and test my model! 

## Credits

This project is adapted from the example described by **Mateus Dias Ribeiro** and collaborators in [DeepCFD](https://arxiv.org/abs/2004.08826). In the present project, we have improved upon their model and added a **Physics informed constraint** to the simulation data.
