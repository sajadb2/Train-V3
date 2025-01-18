# GPT-2 Model Training

This repository contains the implementation of a GPT-2 model for text generation tasks. The model is designed to be trained from scratch or fine-tuned on specific datasets.

## Model Architecture

The model is based on the GPT-2 architecture, which is a transformer-based decoder model. The current configuration is set to achieve approximately 124 million parameters.

### Configuration Parameters

- **Block Size**: 1024 (maximum sequence length)
- **Vocabulary Size**: 50176 (number of tokens)
- **Number of Layers**: 12 (transformer blocks)
- **Number of Heads**: 8 (attention heads)
- **Embedding Dimension**: 768 (size of the embedding vectors)

### Causal Self-Attention

The model utilizes a causal self-attention mechanism, which allows it to generate text by attending to previous tokens in the sequence.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
```

## Installation

To run this model, ensure you have the following dependencies installed:

```bash
pip install torch transformers tiktoken
```

## Training the Model

To train the model, run the `train_model_Transformer.py` script. The script includes a training loop that processes batches of data and updates the model weights based on the calculated loss.

### Usage

1. Prepare your dataset and ensure it is in the correct format.
2. Adjust the hyperparameters in the `train_model_Transformer.py` file as needed (e.g., learning rate, batch size).
3. Run the training script:

```bash
python train_model_Transformer.py
```

## Monitoring Loss

The training loop prints the loss at each step. You can adjust the learning rate and other hyperparameters to help reduce the loss below 0.09.

## License

None

## Acknowledgments

This implementation is inspired by the original GPT-2 model developed by OpenAI. For more information, visit the [OpenAI GPT-2 GitHub repository](https://github.com/openai/gpt-2).

## Tags

- GPT-2
- Text Generation
- Machine Learning
- Natural Language Processing
- PyTorch
- Transformers
- Deep Learning
