/** A Transformer block: self-attention followed by feed-forward, each with residual add & norm. */
class TransformerBlock {
    private int embedDim;
    MultiHeadAttention selfAttention;
    FeedForwardNetwork feedForward;
    LayerNorm norm1;
    LayerNorm norm2;
    // Stored intermediate results for backward
    private double[][] lastInput;
    private double[][] lastRes1;
    private double[][] lastNorm1;

    public TransformerBlock(int embedDim, int numHeads, int hiddenDim) {
        this.embedDim = embedDim;
        // Initialize sub-layers
        selfAttention = new MultiHeadAttention(embedDim, numHeads);
        feedForward = new FeedForwardNetwork(embedDim, hiddenDim);
        norm1 = new LayerNorm(embedDim);
        norm2 = new LayerNorm(embedDim);
    }

    /** Forward pass through the transformer block. */
    public double[][] forward(double[][] input) {
        lastInput = input;
        // 1. Multi-head attention (decoder uses masked self-attention)
        double[][] attnOut = selfAttention.forward(input);
        // 2. First residual connection: add attention output to input
        lastRes1 = new double[input.length][embedDim];
        for (int t = 0; t < input.length; t++) {
            for (int j = 0; j < embedDim; j++) {
                lastRes1[t][j] = input[t][j] + attnOut[t][j];
            }
        }
        // 3. Layer normalization after attention
        lastNorm1 = norm1.forward(lastRes1);
        // 4. Feed-forward network
        double[][] ffOut = feedForward.forward(lastNorm1);
        // 5. Second residual: add feed-forward output to norm1 output
        double[][] res2 = new double[lastNorm1.length][embedDim];
        for (int t = 0; t < lastNorm1.length; t++) {
            for (int j = 0; j < embedDim; j++) {
                res2[t][j] = lastNorm1[t][j] + ffOut[t][j];
            }
        }
        // 6. Layer normalization after feed-forward
        double[][] output = norm2.forward(res2);
        return output;
    }

    /** Backward pass through the transformer block. Returns gradient w.rt block input. */
    public double[][] backward(double[][] gradOutput) {
        int seqLen = gradOutput.length;
        // Backprop through second layer norm
        double[][] dRes2 = norm2.backward(gradOutput);
        // Split grad for second residual connection (to norm1 output and FFN output)
        double[][] dNorm1_from_res = new double[seqLen][embedDim];
        double[][] dFfOut = new double[seqLen][embedDim];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < embedDim; j++) {
                dNorm1_from_res[t][j] = dRes2[t][j];
                dFfOut[t][j] = dRes2[t][j];
            }
        }
        // Backprop through feed-forward network
        double[][] dNorm1_from_ff = feedForward.backward(dFfOut);
        // Sum gradients from both paths for norm1 output
        double[][] dNorm1_total = new double[seqLen][embedDim];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < embedDim; j++) {
                dNorm1_total[t][j] = dNorm1_from_res[t][j] + dNorm1_from_ff[t][j];
            }
        }
        // Backprop through first layer norm
        double[][] dRes1 = norm1.backward(dNorm1_total);
        // Split grad for first residual connection (to input and attention output)
        double[][] dInput_from_res = new double[seqLen][embedDim];
        double[][] dAttnOut = new double[seqLen][embedDim];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < embedDim; j++) {
                dInput_from_res[t][j] = dRes1[t][j];
                dAttnOut[t][j] = dRes1[t][j];
            }
        }
        // Backprop through multi-head attention
        double[][] dInput_from_attn = selfAttention.backward(dAttnOut);
        // Sum gradients for block input from both residual paths
        double[][] dInput = new double[seqLen][embedDim];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < embedDim; j++) {
                dInput[t][j] = dInput_from_res[t][j] + dInput_from_attn[t][j];
            }
        }
        return dInput;
    }

    /** Update all sub-layer parameters. */
    public void updateWeights(double learningRate) {
        selfAttention.updateWeights(learningRate);
        feedForward.updateWeights(learningRate);
        norm1.updateWeights(learningRate);
        norm2.updateWeights(learningRate);
    }
}