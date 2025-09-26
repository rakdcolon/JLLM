import java.util.*;

/** Multi-head self-attention layer. */
class MultiHeadAttention {
    private int embedDim;
    private int numHeads;
    private int headDim;
    // Weight matrices for Q, K, V for each head (dimensions: [embedDim x headDim] each)
    double[][][] Wq;
    double[][][] Wk;
    double[][][] Wv;
    // Output weight matrix Wo of shape [(numHeads*headDim) x embedDim] (equals [embedDim x embedDim])
    double[][] Wo;
    // Gradient matrices for weights
    double[][][] gradWq;
    double[][][] gradWk;
    double[][][] gradWv;
    double[][] gradWo;
    // Stored values from forward pass for backpropagation
    private double[][] lastInput;          // [seqLen x embedDim]
    private double[][][] lastQ;            // [numHeads x seqLen x headDim]
    private double[][][] lastK;            // [numHeads x seqLen x headDim]
    private double[][][] lastV;            // [numHeads x seqLen x headDim]
    private double[][][] lastSoftmax;      // [numHeads x seqLen x seqLen] (attention weights after softmax)
    private double[][] lastCombined;       // [seqLen x (numHeads*headDim)] concatenated head outputs (before Wo)

    public MultiHeadAttention(int embedDim, int numHeads) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.headDim = embedDim / numHeads;
        // Initialize weight matrices with small random values
        Wq = new double[numHeads][embedDim][headDim];
        Wk = new double[numHeads][embedDim][headDim];
        Wv = new double[numHeads][embedDim][headDim];
        Wo = new double[numHeads * headDim][embedDim];  // shape [embedDim x embedDim]
        gradWq = new double[numHeads][embedDim][headDim];
        gradWk = new double[numHeads][embedDim][headDim];
        gradWv = new double[numHeads][embedDim][headDim];
        gradWo = new double[numHeads * headDim][embedDim];
        Random rand = new Random(1);
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < embedDim; i++) {
                for (int j = 0; j < headDim; j++) {
                    Wq[h][i][j] = rand.nextDouble() * 0.2 - 0.1;
                    Wk[h][i][j] = rand.nextDouble() * 0.2 - 0.1;
                    Wv[h][i][j] = rand.nextDouble() * 0.2 - 0.1;
                }
            }
        }
        for (int i = 0; i < numHeads * headDim; i++) {
            for (int j = 0; j < embedDim; j++) {
                Wo[i][j] = rand.nextDouble() * 0.2 - 0.1;
            }
        }
    }

    /** Forward pass for multi-head self-attention (with causal mask for autoregressive modeling). 
     * Input: [seqLen x embedDim], Output: [seqLen x embedDim]. */
    public double[][] forward(double[][] input) {
        int seqLen = input.length;
        lastInput = input;
        // Initialize storage for Q, K, V, and attention outputs
        lastQ = new double[numHeads][seqLen][headDim];
        lastK = new double[numHeads][seqLen][headDim];
        lastV = new double[numHeads][seqLen][headDim];
        lastSoftmax = new double[numHeads][seqLen][seqLen];
        lastCombined = new double[seqLen][numHeads * headDim];
        double[][] output = new double[seqLen][embedDim];

        // Compute Q, K, V for each head
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLen; i++) {
                // Calculate Q_h[i], K_h[i], V_h[i] by multiplying input[i] by Wq[h], Wk[h], Wv[h]
                for (int j = 0; j < headDim; j++) {
                    double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                    for (int k = 0; k < embedDim; k++) {
                        double inp = input[i][k];
                        sumQ += inp * Wq[h][k][j];
                        sumK += inp * Wk[h][k][j];
                        sumV += inp * Wv[h][k][j];
                    }
                    lastQ[h][i][j] = sumQ;
                    lastK[h][i][j] = sumK;
                    lastV[h][i][j] = sumV;
                }
            }
            // Scaled dot-product attention for head h
            double invSqrtDim = 1.0 / Math.sqrt(headDim);
            for (int i = 0; i < seqLen; i++) {
                // Compute attention scores for query position i against all key positions j
                double[] scores = new double[seqLen];
                for (int j = 0; j < seqLen; j++) {
                    // Dot product: Q_h[i] · K_h[j]
                    double score = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        score += lastQ[h][i][d] * lastK[h][j][d];
                    }
                    scores[j] = score * invSqrtDim; // scale by 1/sqrt(headDim)
                }
                // Apply causal mask: for j > i, prevent attention (will be zero after softmax)
                double maxScore = Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= i; j++) {
                    if (scores[j] > maxScore) {
                        maxScore = scores[j];
                    }
                }
                // Softmax calculation (only over j <= i)
                double sumExp = 0.0;
                for (int j = 0; j <= i; j++) {
                    double expVal = Math.exp(scores[j] - maxScore);
                    sumExp += expVal;
                    lastSoftmax[h][i][j] = expVal;  // store numerator temporarily
                }
                for (int j = 0; j <= i; j++) {
                    lastSoftmax[h][i][j] /= sumExp;  // normalize to get probability
                }
                // For j > i, lastSoftmax remains 0 (masked out).
                // Compute attention output for position i (weighted sum of V_h[j] for j <= i)
                for (int d = 0; d < headDim; d++) {
                    double sum = 0.0;
                    for (int j = 0; j <= i; j++) {
                        sum += lastSoftmax[h][i][j] * lastV[h][j][d];
                    }
                    lastCombined[i][h * headDim + d] = sum;
                }
            } // end for each query i
        } // end for each head

        // Final linear projection: combine all head outputs via Wo
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < embedDim; j++) {
                double sum = 0.0;
                // Note: numHeads * headDim == embedDim
                for (int k = 0; k < numHeads * headDim; k++) {
                    sum += lastCombined[i][k] * Wo[k][j];
                }
                output[i][j] = sum;
            }
        }
        return output;
    }

    /** Backward pass for multi-head attention. 
     * gradOutput: gradient of loss w.rt this layer's output (shape [seqLen x embedDim]). 
     * Returns gradient w.rt input (shape [seqLen x embedDim]). */
    public double[][] backward(double[][] gradOutput) {
        int seqLen = lastInput.length;
        // Initialize gradient for input
        double[][] gradInput = new double[seqLen][embedDim];

        // Gradients for output projection Wo
        // gradWo = (lastCombined)^T * gradOutput
        for (int i = 0; i < numHeads * headDim; i++) {
            for (int j = 0; j < embedDim; j++) {
                double sum = 0.0;
                for (int t = 0; t < seqLen; t++) {
                    sum += lastCombined[t][i] * gradOutput[t][j];
                }
                gradWo[i][j] += sum;
            }
        }
        // Gradient of concatenated head outputs (before Wo): dCombined = gradOutput * Wo^T
        double[][] gradCombined = new double[seqLen][numHeads * headDim];
        for (int t = 0; t < seqLen; t++) {
            for (int i = 0; i < numHeads * headDim; i++) {
                double sum = 0.0;
                for (int j = 0; j < embedDim; j++) {
                    sum += gradOutput[t][j] * Wo[i][j];
                }
                gradCombined[t][i] = sum;
            }
        }

        // Backpropagate into each head
        for (int h = 0; h < numHeads; h++) {
            // Gradient of head output (slice corresponding to this head from gradCombined)
            double[][] dHeadOut = new double[seqLen][headDim];
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(gradCombined[t], h * headDim, dHeadOut[t], 0, headDim);
            }

            // Gradients for softmax output and V: dV = S^T * dHeadOut, dS = dHeadOut * V^T
            double[][] dS = new double[seqLen][seqLen];
            double[][] dV = new double[seqLen][headDim];
            for (int j = 0; j < seqLen; j++) {
                // dV[j] = sum_{i} softmax[i,j] * dHeadOut[i]
                for (int d = 0; d < headDim; d++) {
                    double sum = 0.0;
                    for (int i = 0; i < seqLen; i++) {
                        sum += lastSoftmax[h][i][j] * dHeadOut[i][d];
                    }
                    dV[j][d] = sum;
                }
            }
            for (int i = 0; i < seqLen; i++) {
                // Only j <= i had nonzero softmax
                for (int j = 0; j <= i; j++) {
                    // dS[i,j] = dHeadOut[i] ⋅ V[j]^T
                    double sum = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        sum += dHeadOut[i][d] * lastV[h][j][d];
                    }
                    dS[i][j] = sum;
                }
            }
            // Backprop through softmax to get gradient of scores (pre-softmax)
            double[][] dScore = new double[seqLen][seqLen];
            for (int i = 0; i < seqLen; i++) {
                double sumRow = 0.0;
                for (int j = 0; j <= i; j++) {
                    sumRow += dS[i][j] * lastSoftmax[h][i][j];
                }
                for (int j = 0; j <= i; j++) {
                    // gradient of softmax: p * (g - sum(p*g))
                    dScore[i][j] = lastSoftmax[h][i][j] * (dS[i][j] - sumRow);
                }
            }
            // Remove scaling: score = Q*K^T / sqrt(headDim), so d(Q*K^T) = dScore * (1/√headDim)
            double invSqrtDim = 1.0 / Math.sqrt(headDim);
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j <= i; j++) {  // j>i are zero and remain zero
                    dScore[i][j] *= invSqrtDim;
                }
            }
            // Gradients for Q and K: dQ = dScore * K, dK = dScore^T * Q
            double[][] dQ = new double[seqLen][headDim];
            double[][] dK = new double[seqLen][headDim];
            for (int i = 0; i < seqLen; i++) {
                for (int d = 0; d < headDim; d++) {
                    double sumQ = 0.0;
                    double sumK_grad = 0.0;
                    for (int j = 0; j < seqLen; j++) {
                        if (dScore[i][j] != 0.0) {
                            sumQ += dScore[i][j] * lastK[h][j][d];
                        }
                        if (dScore[j][i] != 0.0) {
                            sumK_grad += dScore[j][i] * lastQ[h][j][d];
                        }
                    }
                    dQ[i][d] = sumQ;
                    dK[i][d] = sumK_grad;
                }
            }
            // Gradients for weight matrices Wq, Wk, Wv, and accumulate input grad
            for (int i = 0; i < embedDim; i++) {
                for (int d = 0; d < headDim; d++) {
                    double gradWq_val = 0.0;
                    double gradWk_val = 0.0;
                    double gradWv_val = 0.0;
                    for (int t = 0; t < seqLen; t++) {
                        gradWq_val += lastInput[t][i] * dQ[t][d];
                        gradWk_val += lastInput[t][i] * dK[t][d];
                        gradWv_val += lastInput[t][i] * dV[t][d];
                    }
                    gradWq[h][i][d] += gradWq_val;
                    gradWk[h][i][d] += gradWk_val;
                    gradWv[h][i][d] += gradWv_val;
                }
            }
            // Gradient w.rt input (accumulate from Q, K, V paths)
            for (int t = 0; t < seqLen; t++) {
                for (int i = 0; i < embedDim; i++) {
                    // dX = dQ * Wq^T + dK * Wk^T + dV * Wv^T
                    double gradXi = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        gradXi += dQ[t][d] * Wq[h][i][d];
                        gradXi += dK[t][d] * Wk[h][i][d];
                        gradXi += dV[t][d] * Wv[h][i][d];
                    }
                    gradInput[t][i] += gradXi;
                }
            }
        } // end for each head

        return gradInput;
    }

    /** Update weights using accumulated gradients, then reset gradients to zero. */
    public void updateWeights(double learningRate) {
        // Update Q, K, V weight matrices
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < embedDim; i++) {
                for (int j = 0; j < headDim; j++) {
                    Wq[h][i][j] -= learningRate * gradWq[h][i][j];
                    Wk[h][i][j] -= learningRate * gradWk[h][i][j];
                    Wv[h][i][j] -= learningRate * gradWv[h][i][j];
                    gradWq[h][i][j] = 0.0;
                    gradWk[h][i][j] = 0.0;
                    gradWv[h][i][j] = 0.0;
                }
            }
        }
        // Update output projection Wo
        for (int i = 0; i < numHeads * headDim; i++) {
            for (int j = 0; j < embedDim; j++) {
                Wo[i][j] -= learningRate * gradWo[i][j];
                gradWo[i][j] = 0.0;
            }
        }
    }
}