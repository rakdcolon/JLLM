import java.util.*;

/** The overall Transformer model (GPT-2 style). Assembles embedding, transformer blocks, and output projection. */
class TransformerModel {
    private int vocabSize;
    private int embedDim;
    private int numHeads;
    private int numLayers;
    private int hiddenDim;
    EmbeddingLayer embedding;
    TransformerBlock[] blocks;
    // Output layer weights (embedDim -> vocabSize) and bias
    double[][] Wout;
    double[] bout;
    // Gradient accumulators for output layer
    double[][] gradWout;
    double[] gradBout;
    // Store last hidden state from forward pass (to use in backward for output grad calculation)
    private double[][] lastHidden;

    public TransformerModel(int vocabSize, int embedDim, int numHeads, int numLayers, int hiddenDim, int maxSeqLen) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.hiddenDim = hiddenDim;
        // Initialize embedding and blocks
        embedding = new EmbeddingLayer(vocabSize, embedDim, maxSeqLen);
        blocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            blocks[i] = new TransformerBlock(embedDim, numHeads, hiddenDim);
        }
        // Initialize output projection weights
        Wout = new double[embedDim][vocabSize];
        bout = new double[vocabSize];
        gradWout = new double[embedDim][vocabSize];
        gradBout = new double[vocabSize];
        Random rand = new Random(3);
        for (int i = 0; i < embedDim; i++) {
            for (int j = 0; j < vocabSize; j++) {
                Wout[i][j] = rand.nextDouble() * 0.2 - 0.1;
            }
        }
        for (int j = 0; j < vocabSize; j++) {
            bout[j] = 0.0;
        }
    }

    /** Forward pass: computes logits (unnormalized log-probabilities) for each position in the input sequence. */
    public double[][] forward(int[] inputTokens) {
        // 1. Embedding
        double[][] x = embedding.forward(inputTokens);
        // 2. Transformer blocks
        for (int i = 0; i < numLayers; i++) {
            x = blocks[i].forward(x);
        }
        lastHidden = x;  // store final hidden state
        // 3. Output layer: compute logits = hidden * Wout + bout
        int seqLen = x.length;
        double[][] logits = new double[seqLen][vocabSize];
        for (int t = 0; t < seqLen; t++) {
            for (int v = 0; v < vocabSize; v++) {
                double sum = bout[v];
                for (int j = 0; j < embedDim; j++) {
                    sum += x[t][j] * Wout[j][v];
                }
                logits[t][v] = sum;
            }
        }
        return logits;
    }

    /** Backward pass: given gradient of loss w.rt logits (same shape as logits), compute all gradients. */
    public void backward(double[][] gradLogits) {
        int seqLen = gradLogits.length;
        // Gradients for output weights and bias
        for (int j = 0; j < embedDim; j++) {
            for (int v = 0; v < vocabSize; v++) {
                double sum = 0.0;
                for (int t = 0; t < seqLen; t++) {
                    sum += lastHidden[t][j] * gradLogits[t][v];
                }
                gradWout[j][v] += sum;
            }
        }
        for (int v = 0; v < vocabSize; v++) {
            double sum = 0.0;
            for (int t = 0; t < seqLen; t++) {
                sum += gradLogits[t][v];
            }
            gradBout[v] += sum;
        }
        // Gradient w.rt final hidden state
        double[][] gradHidden = new double[seqLen][embedDim];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < embedDim; j++) {
                double sum = 0.0;
                for (int v = 0; v < vocabSize; v++) {
                    sum += gradLogits[t][v] * Wout[j][v];
                }
                gradHidden[t][j] = sum;
            }
        }
        // Backprop through each transformer block (in reverse order)
        double[][] gradX = gradHidden;
        for (int i = numLayers - 1; i >= 0; i--) {
            gradX = blocks[i].backward(gradX);
        }
        // Backprop through embedding layer
        embedding.backward(gradX);
    }

    /** Update all model parameters using accumulated gradients (then reset grads to zero). */
    public void updateWeights(double learningRate) {
        // Update output layer weights
        for (int j = 0; j < embedDim; j++) {
            for (int v = 0; v < vocabSize; v++) {
                Wout[j][v] -= learningRate * gradWout[j][v];
                gradWout[j][v] = 0.0;
            }
        }
        for (int v = 0; v < vocabSize; v++) {
            bout[v] -= learningRate * gradBout[v];
            gradBout[v] = 0.0;
        }
        // Update transformer blocks and embedding
        for (int i = 0; i < numLayers; i++) {
            blocks[i].updateWeights(learningRate);
        }
        embedding.updateWeights(learningRate);
    }
}