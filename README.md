# Train-V3: MNIST Digit Classification with PyTorch

    A simple Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.

## Project Description
    This project implements a lightweight CNN architecture to classify handwritten digits from the MNIST dataset. The model achieves >97% accuracy on the test set with minimal training.

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
   
### Testing
    Test the trained model using:
        python test.py

Performance metrics
File structure
    The project includes the following files:
        Train-V3/
        ├── train.py # Training script
        ├── test.py # Testing script
        ├── model.pth # Saved model weights
        ├── requirements.txt
        └── README.md

Other standard documentation sections
You should customize:
The repository URL
License information
Your name in the Authors section
Any specific acknowledgments
Additional sections based on your project's needs
You can also add:
Screenshots of training progress
Example predictions
More detailed performance metrics
Hardware requirements
Known issues or limitations
Future improvements planned
