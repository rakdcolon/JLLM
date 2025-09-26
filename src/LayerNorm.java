/** Layer Normalization layer. Normalizes input across the embedding dimension for each position. */
class LayerNorm {
    private int dim;
    // Learnable parameters
    double[] gamma;
    double[] beta;
    // Gradient accumulators for parameters
    double[] gradGamma;
    double[] gradBeta;
    // Stored values from forward pass
    private double[][] lastInput;
    private double[][] lastNormalized;
    private double[] lastMean;
    private double[] lastInvVar;
    private double epsilon;

    public LayerNorm(int dim) {
        this.dim = dim;
        this.gamma = new double[dim];
        this.beta = new double[dim];
        this.gradGamma = new double[dim];
        this.gradBeta = new double[dim];
        this.epsilon = 1e-5;
        // Initialize gamma = 1 (scale), beta = 0 (shift)
        for (int i = 0; i < dim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    /** Forward pass: normalize each row of input to mean 0 and variance 1, then scale by gamma and shift by beta. */
    public double[][] forward(double[][] input) {
        int seqLen = input.length;
        lastInput = input;
        lastNormalized = new double[seqLen][dim];
        lastMean = new double[seqLen];
        lastInvVar = new double[seqLen];
        double[][] output = new double[seqLen][dim];
        for (int t = 0; t < seqLen; t++) {
            // Compute mean for this position (over all features)
            double mean = 0.0;
            for (int j = 0; j < dim; j++) {
                mean += input[t][j];
            }
            mean /= dim;
            // Compute variance
            double var = 0.0;
            for (int j = 0; j < dim; j++) {
                double diff = input[t][j] - mean;
                var += diff * diff;
            }
            var /= dim;
            double invStd = 1.0 / Math.sqrt(var + epsilon);  // 1/sigma
            lastMean[t] = mean;
            lastInvVar[t] = invStd;
            // Normalize and apply gamma and beta
            for (int j = 0; j < dim; j++) {
                double normalized = (input[t][j] - mean) * invStd;
                lastNormalized[t][j] = normalized;
                output[t][j] = gamma[j] * normalized + beta[j];
            }
        }
        return output;
    }

    /** Backward pass: gradOutput [seqLen x dim] -> returns gradient [seqLen x dim] w.rt input. */
    public double[][] backward(double[][] gradOutput) {
        int seqLen = gradOutput.length;
        double[][] gradInput = new double[seqLen][dim];
        // Gradients for gamma and beta parameters
        for (int j = 0; j < dim; j++) {
            double sumGradGamma = 0.0;
            double sumGradBeta = 0.0;
            for (int t = 0; t < seqLen; t++) {
                sumGradGamma += gradOutput[t][j] * lastNormalized[t][j];
                sumGradBeta += gradOutput[t][j];
            }
            gradGamma[j] += sumGradGamma;
            gradBeta[j] += sumGradBeta;
        }
        // Gradient w.rt input
        for (int t = 0; t < seqLen; t++) {
            double invVar = lastInvVar[t];
            // Compute sum of gradients and sum of gradient * (x - mean)
            double sum_dNorm = 0.0;
            double sum_dNorm_norm = 0.0;
            for (int j = 0; j < dim; j++) {
                // dNormalized_j = gradOutput_j * gamma_j
                double dNormalized = gradOutput[t][j] * gamma[j];
                sum_dNorm += dNormalized;
                sum_dNorm_norm += dNormalized * lastNormalized[t][j];
            }
            // Now compute gradient for each input feature j
            for (int j = 0; j < dim; j++) {
                double dNormalized = gradOutput[t][j] * gamma[j];
                // Using formula: gradInput_j = (1/dim) * invVar * [dim * dNormalized_j - sum(dNormalized) - lastNormalized_j * sum(dNormalized * lastNormalized)]
                gradInput[t][j] = (1.0 / dim) * invVar * ((dim * dNormalized) - sum_dNorm - lastNormalized[t][j] * sum_dNorm_norm);
            }
        }
        return gradInput;
    }

    /** Update parameters gamma and beta using accumulated gradients (and then reset grads). */
    public void updateWeights(double learningRate) {
        for (int j = 0; j < dim; j++) {
            gamma[j] -= learningRate * gradGamma[j];
            beta[j] -= learningRate * gradBeta[j];
            gradGamma[j] = 0.0;
            gradBeta[j] = 0.0;
        }
    }
}