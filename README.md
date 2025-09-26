# JLLM

A Java implementation of a Large Language Model with BPE (Byte Pair Encoding) tokenization and Transformer architecture.

## Features

- **BPE Tokenizer**: Implements Byte Pair Encoding for efficient text tokenization
- **Transformer Model**: GPT-2 style transformer with multi-head attention, feed-forward networks, and layer normalization
- **File Input Support**: Can train on custom text files or use a default corpus
- **Training Loop**: Includes forward/backward pass with cross-entropy loss and gradient descent

## Usage

### Compile the project:
```bash
javac src/*.java
```

### Run with default corpus:
```bash
java -cp src Main
```

### Run with custom text file:
```bash
java -cp src Main <path-to-text-file>
```

### Examples:
```bash
# Using a local text file
java -cp src Main sample_text.txt

# Using an absolute path
java -cp src Main /path/to/your/document.txt
```

## Text File Requirements

- The program accepts any UTF-8 encoded text file
- Larger files generally work better with the BPE tokenizer
- The program automatically converts text to lowercase for training
- If the file cannot be read, the program falls back to a default corpus

## Architecture

- **Main.java**: Entry point with training loop and file handling
- **BPETokenizer.java**: Byte Pair Encoding implementation
- **TransformerModel.java**: Main transformer model
- **TransformerBlock.java**: Individual transformer block with attention and FFN
- **MultiheadAttention.java**: Multi-head self-attention mechanism
- **FeedForwardNetwork.java**: Feed-forward network component
- **EmbeddingLayer.java**: Token and positional embeddings
- **LayerNorm.java**: Layer normalization implementation
