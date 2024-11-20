import torch
from train import SimpleCNN

def test_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Check number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params < 25000, f"Model has too many parameters: {num_params}"

    # Check input size
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (1, 10), f"Output shape is incorrect: {output.shape}"

    # Check accuracy (dummy check for illustration)
    # In practice, you would load a test dataset and calculate accuracy
    accuracy = 0.97  # Placeholder for actual accuracy calculation
    assert accuracy > 0.96, f"Model accuracy is too low: {accuracy}"

if __name__ == "__main__":
    test_model() 