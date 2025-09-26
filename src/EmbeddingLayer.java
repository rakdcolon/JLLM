import java.util.*;

/** Embedding layer for token and positional embeddings. */
class EmbeddingLayer {
    private int vocabSize;
    private int embedDim;
    private int maxSeqLen;
    /** Token embedding matrix: vocabSize x embedDim */
    double[][] tokenEmbedding;
    /** Positional embedding matrix: maxSeqLen x embedDim */
    double[][] positionalEmbedding;
    /** Gradient for token embedding matrix (same size) */
    double[][] gradTokenEmbedding;
    /** Gradient for positional embedding matrix (same size as positionalEmbedding) */
    double[][] gradPositionalEmbedding;
    /** Last input token indices (stored for backward pass) */
    private int[] lastInputTokens;

    public EmbeddingLayer(int vocabSize, int embedDim, int maxSeqLen) {
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        // Initialize embeddings with small random values
        tokenEmbedding = new double[vocabSize][embedDim];
        positionalEmbedding = new double[maxSeqLen][embedDim];
        gradTokenEmbedding = new double[vocabSize][embedDim];
        gradPositionalEmbedding = new double[maxSeqLen][embedDim];
        Random rand = new Random(0); // fixed seed for reproducibility
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embedDim; j++) {
                tokenEmbedding[i][j] = rand.nextDouble() * 0.2 - 0.1;
            }
        }
        for (int i = 0; i < maxSeqLen; i++) {
            for (int j = 0; j < embedDim; j++) {
                positionalEmbedding[i][j] = rand.nextDouble() * 0.2 - 0.1;
            }
        }
    }

    /** Forward pass: convert an array of token IDs to an array of embedding vectors (token + positional). */
    public double[][] forward(int[] tokenIndices) {
        int seqLen = tokenIndices.length;
        double[][] output = new double[seqLen][embedDim];
        this.lastInputTokens = tokenIndices;
        for (int i = 0; i < seqLen; i++) {
            int tokenId = tokenIndices[i];
            for (int j = 0; j < embedDim; j++) {
                output[i][j] = tokenEmbedding[tokenId][j] + positionalEmbedding[i][j];
            }
        }
        return output;
    }

    /** Backward pass: accumulate gradients for embeddings. 
     * gradOutput: gradient of loss w.rt this layer's output (shape [seqLen x embedDim]). 
     * This updates gradTokenEmbedding and gradPositionalEmbedding. */
    public void backward(double[][] gradOutput) {
        int seqLen = lastInputTokens.length;
        for (int i = 0; i < seqLen; i++) {
            int tokenId = lastInputTokens[i];
            for (int j = 0; j < embedDim; j++) {
                gradTokenEmbedding[tokenId][j] += gradOutput[i][j];
                gradPositionalEmbedding[i][j] += gradOutput[i][j];
            }
        }
        // (No gradient to return for input tokens, since they are indices, not differentiable)
    }

    /** Update embedding weights using accumulated gradients (then reset grads). */
    public void updateWeights(double learningRate) {
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embedDim; j++) {
                tokenEmbedding[i][j] -= learningRate * gradTokenEmbedding[i][j];
                gradTokenEmbedding[i][j] = 0.0;
            }
        }
        for (int i = 0; i < maxSeqLen; i++) {
            for (int j = 0; j < embedDim; j++) {
                positionalEmbedding[i][j] -= learningRate * gradPositionalEmbedding[i][j];
                gradPositionalEmbedding[i][j] = 0.0;
            }
        }
    }
}