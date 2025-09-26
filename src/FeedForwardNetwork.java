import java.util.*;

/** Feed-forward network layer (position-wise) used in transformer blocks. */
class FeedForwardNetwork {
    private int inputDim;
    private int hiddenDim;
    // Weights and biases
    double[][] W1;  // [inputDim x hiddenDim]
    double[] b1;    // [hiddenDim]
    double[][] W2;  // [hiddenDim x inputDim]
    double[] b2;    // [inputDim]
    // Gradient accumulators
    double[][] gradW1;
    double[] gradb1;
    double[][] gradW2;
    double[] gradb2;
    // Stored values from forward pass for backprop
    private double[][] lastInput;
    private double[][] lastHiddenPre;  // [seqLen x hiddenDim] pre-activation (before ReLU)

    public FeedForwardNetwork(int inputDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        W1 = new double[inputDim][hiddenDim];
        b1 = new double[hiddenDim];
        W2 = new double[hiddenDim][inputDim];
        b2 = new double[inputDim];
        gradW1 = new double[inputDim][hiddenDim];
        gradb1 = new double[hiddenDim];
        gradW2 = new double[hiddenDim][inputDim];
        gradb2 = new double[inputDim];
        Random rand = new Random(2);
        // Initialize weights (small random) and biases (zeros)
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                W1[i][j] = rand.nextDouble() * 0.2 - 0.1;
            }
        }
        for (int j = 0; j < hiddenDim; j++) {
            b1[j] = 0.0;
            for (int i = 0; i < inputDim; i++) {
                W2[j][i] = rand.nextDouble() * 0.2 - 0.1;
            }
        }
        for (int i = 0; i < inputDim; i++) {
            b2[i] = 0.0;
        }
    }

    /** Forward pass: input [seqLen x inputDim] -> output [seqLen x inputDim]. */
    public double[][] forward(double[][] input) {
        int seqLen = input.length;
        lastInput = input;
        lastHiddenPre = new double[seqLen][hiddenDim];
        double[][] hiddenPost = new double[seqLen][hiddenDim];
        double[][] output = new double[seqLen][inputDim];
        // First linear layer + ReLU
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < hiddenDim; j++) {
                double sum = b1[j];
                for (int i = 0; i < inputDim; i++) {
                    sum += input[t][i] * W1[i][j];
                }
                lastHiddenPre[t][j] = sum;
                // Apply ReLU activation
                hiddenPost[t][j] = (sum > 0) ? sum : 0.0;
            }
        }
        // Second linear layer
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < inputDim; i++) {
                double sum = b2[i];
                for (int j = 0; j < hiddenDim; j++) {
                    sum += hiddenPost[t][j] * W2[j][i];
                }
                output[t][i] = sum;
            }
        }
        return output;
    }

    /** Backward pass: gradOutput [seqLen x inputDim] -> returns gradInput [seqLen x inputDim]. */
    public double[][] backward(double[][] gradOutput) {
        int seqLen = gradOutput.length;
        double[][] gradInput = new double[seqLen][inputDim];
        // Gradients for W2 and b2
        for (int j = 0; j < hiddenDim; j++) {
            for (int i = 0; i < inputDim; i++) {
                double sumGradW2 = 0.0;
                for (int t = 0; t < seqLen; t++) {
                    // ReLU output = max(0, lastHiddenPre), reuse lastHiddenPre to get post-activation
                    double hiddenPost = (lastHiddenPre[t][j] > 0) ? lastHiddenPre[t][j] : 0.0;
                    sumGradW2 += hiddenPost * gradOutput[t][i];
                }
                gradW2[j][i] += sumGradW2;
            }
        }
        for (int i = 0; i < inputDim; i++) {
            double sumGradB2 = 0.0;
            for (int t = 0; t < seqLen; t++) {
                sumGradB2 += gradOutput[t][i];
            }
            gradb2[i] += sumGradB2;
        }
        // Gradient of hidden layer (after activation)
        double[][] gradHidden = new double[seqLen][hiddenDim];
        for (int t = 0; t < seqLen; t++) {
            for (int j = 0; j < hiddenDim; j++) {
                // dHiddenPost = sum_i gradOutput[i] * W2[j][i]
                double sum = 0.0;
                for (int i = 0; i < inputDim; i++) {
                    sum += gradOutput[t][i] * W2[j][i];
                }
                // Backprop through ReLU: if pre-activation <= 0, gradient becomes 0
                gradHidden[t][j] = (lastHiddenPre[t][j] > 0) ? sum : 0.0;
            }
        }
        // Gradients for W1 and b1, and compute gradInput
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                double sumGradW1 = 0.0;
                for (int t = 0; t < seqLen; t++) {
                    sumGradW1 += lastInput[t][i] * gradHidden[t][j];
                }
                gradW1[i][j] += sumGradW1;
            }
        }
        for (int j = 0; j < hiddenDim; j++) {
            double sumGradB1 = 0.0;
            for (int t = 0; t < seqLen; t++) {
                sumGradB1 += gradHidden[t][j];
            }
            gradb1[j] += sumGradB1;
        }
        // gradInput = gradHidden * W1^T
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < inputDim; i++) {
                double sum = 0.0;
                for (int j = 0; j < hiddenDim; j++) {
                    sum += gradHidden[t][j] * W1[i][j];
                }
                gradInput[t][i] = sum;
            }
        }
        return gradInput;
    }

    /** Update weights using accumulated gradients (and reset them to zero). */
    public void updateWeights(double learningRate) {
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                W1[i][j] -= learningRate * gradW1[i][j];
                gradW1[i][j] = 0.0;
            }
        }
        for (int j = 0; j < hiddenDim; j++) {
            b1[j] -= learningRate * gradb1[j];
            gradb1[j] = 0.0;
        }
        for (int j = 0; j < hiddenDim; j++) {
            for (int i = 0; i < inputDim; i++) {
                W2[j][i] -= learningRate * gradW2[j][i];
                gradW2[j][i] = 0.0;
            }
        }
        for (int i = 0; i < inputDim; i++) {
            b2[i] -= learningRate * gradb2[i];
            gradb2[i] = 0.0;
        }
    }
}