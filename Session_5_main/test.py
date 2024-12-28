import torch
import unittest
from train import SimpleCNN, calculate_accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TestMNISTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SimpleCNN()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model.load_state_dict(torch.load('model.pth', map_location=cls.device))
        cls.model.eval()
        
        # Load test data
        transform = transforms.Compose([transforms.ToTensor()])
        cls.test_dataset = datasets.MNIST(root='./data', train=False, 
                                        transform=transform, download=True)
        cls.test_loader = DataLoader(cls.test_dataset, batch_size=1000, shuffle=False)

    def test_model_architecture(self):
        """Test if model architecture matches expected structure"""
        self.assertTrue(hasattr(self.model, 'conv1'))
        self.assertTrue(hasattr(self.model, 'conv2'))
        self.assertTrue(hasattr(self.model, 'fc1'))
        self.assertEqual(self.model.conv1.out_channels, 8)
        self.assertEqual(self.model.conv2.out_channels, 16)

    def test_model_output_shape(self):
        """Test if model produces correct output shape"""
        sample_input = torch.randn(1, 1, 28, 28)
        output = self.model(sample_input)
        self.assertEqual(output.shape, (1, 10))

    def test_model_accuracy(self):
        """Test if model meets minimum accuracy threshold"""
        accuracy = calculate_accuracy(self.model, self.test_loader, self.device)
        self.assertGreater(accuracy, 95.0)

    def test_model_deterministic(self):
        """Test if model produces consistent outputs"""
        sample_input = torch.randn(1, 1, 28, 28)
        output1 = self.model(sample_input)
        output2 = self.model(sample_input)
        self.assertTrue(torch.allclose(output1, output2))

if __name__ == '__main__':
    unittest.main() 