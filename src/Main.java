import java.util.*;

public class Main {
    public static void main(String[] args) {
        // Sample corpus for training (a simple repetitive text)
        String corpus = "the cat in the hat. the cat is in the hat.";
        corpus = corpus.toLowerCase();  // normalize case
        // Train BPE tokenizer on the corpus
        int vocabSize = 1000;  // target vocabulary size (includes base 256 bytes)
        BPETokenizer tokenizer = new BPETokenizer(vocabSize);
        tokenizer.train(corpus);
        // Encode corpus text to token sequence
        List<Integer> tokenList = tokenizer.encode(corpus);
        int N = tokenList.size();
        // Prepare training data: input tokens and target tokens
        int[] inputTokens = new int[N - 1];
        int[] targetTokens = new int[N - 1];
        for (int i = 0; i < N - 1; i++) {
            inputTokens[i] = tokenList.get(i);
            targetTokens[i] = tokenList.get(i + 1);
        }

        // Initialize Transformer model
        int embedDim = 32;
        int numHeads = 4;
        int numLayers = 2;
        int hiddenDim = 128;
        int maxSeqLen = N;
        TransformerModel model = new TransformerModel(vocabSize, embedDim, numHeads, numLayers, hiddenDim, maxSeqLen);

        // Training loop
        double learningRate = 0.05;
        int epochs = 100;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Forward pass
            double[][] logits = model.forward(inputTokens);
            // Compute loss (cross-entropy) and gradient w.rt logits
            double loss = 0.0;
            double[][] gradLogits = new double[logits.length][logits[0].length];
            for (int t = 0; t < logits.length; t++) {
                // Softmax for token t
                double maxLogit = Double.NEGATIVE_INFINITY;
                for (int v = 0; v < logits[t].length; v++) {
                    if (logits[t][v] > maxLogit) maxLogit = logits[t][v];
                }
                double sumExp = 0.0;
                for (int v = 0; v < logits[t].length; v++) {
                    sumExp += Math.exp(logits[t][v] - maxLogit);
                }
                int target = targetTokens[t];
                // Cross-entropy loss for this position
                double logProbTarget = (logits[t][target] - maxLogit) - Math.log(sumExp);
                loss += -logProbTarget;
                // Gradients for logits (softmax derivative)
                for (int v = 0; v < logits[t].length; v++) {
                    double p = Math.exp(logits[t][v] - maxLogit) / sumExp;
                    gradLogits[t][v] = p;
                    if (v == target) {
                        gradLogits[t][v] -= 1.0;
                    }
                }
            }
            loss /= logits.length;  // average loss per token

            // Backward pass
            model.backward(gradLogits);
            // Update model parameters
            model.updateWeights(learningRate);

            // Print training loss periodically
            if (epoch == 1 || epoch % 10 == 0 || epoch == epochs) {
                System.out.println("Epoch " + epoch + " - Loss: " + loss);
            }
        }
    }
}