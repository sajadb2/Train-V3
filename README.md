# MNIST based model Training & Testing

[![License](https://img.shields.io/github/license/sajadb2/Train-V3)](https://github.com/sajadb2/Train-V3/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/sajadb2/Train-V3)](https://github.com/sajadb2/Train-V3/issues)
[![Stars](https://img.shields.io/github/stars/sajadb2/Train-V3)](https://github.com/sajadb2/Train-V3/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/sajadb2/Train-V3)](https://github.com/sajadb2/Train-V3/commits/main)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

 A simple Convolutional Neural Network (CNN) implementation for MNIST digit 
 classification using PyTorch.

## Project Description
This project implements a lightweight CNN architecture to classify handwritten digits from the MNIST dataset.
The model achieves >95% accuracy on the test set with minimal training. Also augmented an image pycode to just augment the images.

## Model architecture details

    The CNN architecture consists of:
        - 2 Convolutional layers
        - Conv1: 1 → 8 channels (3x3 kernel)
        - Conv2: 8 → 16 channels (3x3 kernel)
        - Max Pooling layers after each convolution
        - Dropout (0.25) for regularization
        - Final Fully Connected layer (16*7*7 → 10)

## Requirements
    bash
    torch
    torchvision
    tqdm
Installation and usage instructions

    1. Clone the repository:
        git clone https://github.com/yourusername/Train-V3.git
        cd Train-V3

    2. Install the dependencies:
        pip install -r requirements.txt

    3. Train the model:
        python train.py        
        Options:
        - `--cpu`: Force CPU usage
        - `--epochs`: Number of training epochs (default: 10)
        - `--batch-size`: Batch size for training (default: 64)
        - `--target-accuracy`: Target accuracy to stop training (default: 96.0)
   
## Testing
    The project includes comprehensive tests:
        - Model architecture validation
        - Output shape verification
        - Accuracy threshold testing
        - Output determinism verification


### File structure
    The project includes the following files:
        Train-V3/
        ├── pycache/                # Pycache directory
        ├── .github/
        │   └── workflows/
        │       └── main.yml           # GitHub Actions workflow configuration
        │       └── python-app.yml     # GitHub Actions workflow configuration
        ├── augmented_images/       # Augmented images directory
        ├── data/
        │   ├── MNIST/
        │   │   ├── raw/
        │   │   │   ├── train-images-idx3-ubyte.gz
        │   │   │   └── train-labels-idx1-ubyte.gz
        │   │   │   ├── t10k-images-idx3-ubyte.gz
        │   │   │   └── t10k-labels-idx1-ubyte.gz
        ├── gitignore              # Git ignore file
        ├── augment.py             # Image augmentation script
        ├── config.py              # Configuration file
        ├── environment.yml        # Environment file
        ├── image.png              # Image file
        ├── model.pth               # Saved model weights
        ├── requirements.txt        # Project dependencies
        ├── test.py                 # Testing script 
        ├── train.py                # Training script
        ├── training_log.txt        # Training log file
        └── README.md              # Project documentation

### Part 2: Image Augmentation
    The project includes image augmentation capabilities:
        - Random rotation (±10 degrees)
        - Random affine transformations
        - Brightness and contrast adjustments
        - Gaussian noise injection
        - Random translations

Run tests using:

<!-- Build Status -->
[![Build Status](https://github.com/sajadb2/Train-V3/workflows/CI/badge.svg)](https://github.com/sajadb2/Train-V3/actions)
    
<!-- Code Coverage -->
[![Coverage Status](https://coveralls.io/repos/github/sajadb2/Train-V3/badge.svg?branch=main)](https://coveralls.io/github/sajadb2/Train-V3?branch=main)

<!-- Dependencies -->
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pytorch.org/vision/stable/index.html)
[![tqdm](https://img.shields.io/badge/tqdm-4.65+-green.svg)](https://tqdm.github.io/)


Screenshots of Augmented Images.
    Refer to augmented_images

The repository URL
    https://github.com/sajadb2/Train-V3.git

License information
    None