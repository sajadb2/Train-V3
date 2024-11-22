import torchvision.transforms as transforms
from PIL import Image
import random
import torch
import matplotlib.pyplot as plt
import os

class ImageAugmenter:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            )
        ])
        
        self.noise_transforms = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(mean=0., std=0.1),
            transforms.ToPILImage()
        ])

class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def augment_and_save_image(image_path, num_augmentations=5, save_dir='augmented_images'):
    """
    Augment a single image multiple times and save/display the results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    augmenter = ImageAugmenter()
    original_image = Image.open(image_path).convert('L')
    
    # Create a figure to display images
    plt.figure(figsize=(15, 3))
    
    # Display original image
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    augmented_images = []
    
    for i in range(num_augmentations):
        # Apply random augmentations
        aug_image = augmenter.transforms(original_image)
        
        # Optionally add noise
        if random.random() > 0.5:
            aug_image = augmenter.noise_transforms(aug_image)
        
        # Save augmented image
        save_path = os.path.join(save_dir, f'augmented_{i+1}.png')
        aug_image.save(save_path)
        
        # Display augmented image
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(aug_image, cmap='gray')
        plt.title(f'Aug {i+1}')
        plt.axis('off')
        
        augmented_images.append(aug_image)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'augmentation_summary.png'))
    plt.show()
    
    return augmented_images

if __name__ == "__main__":
    # Example usage
    image_path = "image.png"  # Make sure this image exists
    augmented_images = augment_and_save_image(image_path, num_augmentations=5)
    print(f"Augmented images saved in 'augmented_images' directory") 