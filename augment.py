import torchvision.transforms as transforms
from PIL import Image
import random

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
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
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

def augment_image(image_path, num_augmentations=5):
    """
    Augment a single image multiple times
    """
    augmenter = ImageAugmenter()
    original_image = Image.open(image_path).convert('L')
    augmented_images = []
    
    for i in range(num_augmentations):
        # Apply random augmentations
        aug_image = augmenter.transforms(original_image)
        
        # Optionally add noise
        if random.random() > 0.5:
            aug_image = augmenter.noise_transforms(aug_image)
            
        augmented_images.append(aug_image)
    
    return augmented_images 