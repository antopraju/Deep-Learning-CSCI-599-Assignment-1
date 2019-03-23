# CSCI-599 Assignment 1

## The objectives of this assignment
* Implement the forward and backward passes as well as the neural network training procedure
* Implement the widely-used optimizers and training tricks including dropout
* Get familiar with TensorFlow by training and designing a network on your own
* Learn how to fine-tune trained networks
* Visualize the learned weights and activation maps of a ConvNet
* Use Grad-CAM to visualize and reason why ConvNet makes certain predictions

## Work on the assignment
Please first clone or download as .zip file of this repository.

Working on the assignment in a virtual environment is highly encouraged.
In this assignment, please use Python `3.5` (or `3.6`).
You will need to make sure that your virtualenv setup is of the correct version of python.

Please see below for executing a virtual environment.
```shell
cd CSCI599-Assignment1
pip3 install virtualenv # If you didn't install it
virtualenv -p $(which python3) /your/path/to/the/virtual/env
source  /your/path/to/the/virtual/env/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# install tensorflow (cpu version, recommended)
pip3 install tensorflow

# install tensorflow (gpu version)
# run this command only if your device supports gpu running
pip3 install tensorflow-gpu

# Work on the assignment
deactivate # Exit the virtual environment
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
python -m ipykernel install --user --name=/your/path/to/the/virtual/env

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=/your/port/

```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Problems
In each of the notebook file, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
Majority of implementations will also be required under `lib` with specified tags.

### Problem 1: Basics of Neural Networks (40 points)
The IPython Notebook `Problem_1.ipynb` will walk you through implementing the basics of neural networks.

### Problem 2: Getting familiar with TensorFlow (25 points)
The IPython Notebook `Problem_2.ipynb` will help you with a better understanding of implementing a simple ConvNet in Tensorflow.

### Problem 3: Training and Fine-tuning on MNIST (10 points)
The IPython Notebook `Problem_3.ipynb` will walk you through training a neural network from scratch on a dataset and fine-tuning on another one for transfer learning.

### Problem 4: Visualizations and CAM (25 points)
The IPython Notebook `Problem_4.ipynb` will gain you insights with what neural networks learn with the skills of visualizing them.

## How to submit

Run the following command to zip all the necessary files for submitting your assignment.

```shell
sh collectSubmission.sh
```

This will create a file named `assignment1.zip`, please rename it with your usc student id (eg. 4916525888.zip), and submit this file through the [Google form](https://goo.gl/forms/RMwyuUxa6V6vz5gF2).
Do NOT create your own .zip file, you might accidentally include non-necessary
materials for grading. We will deduct points if you don't follow the above
submission guideline.

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are
more than welcome to assist.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS (under folder assignment1).
