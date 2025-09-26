import java.util.*;
import java.io.*;
import java.nio.file.*;

public class Main {
    public static void main(String[] args) {
        String corpus;
        
        // Check if a file path is provided as command line argument
        if (args.length > 0) {
            String filePath = args[0];
            try {
                System.out.println("Reading corpus from file: " + filePath);
                File file = new File(filePath);
                long fileSize = file.length();
                
                // Check file size and choose appropriate strategy
                if (fileSize > 50 * 1024 * 1024) { // > 50MB
                    System.out.println("Large file detected (" + fileSize / 1024 / 1024 + "MB). Using chunk-based processing.");
                    corpus = readLargeFileInChunks(filePath);
                } else if (fileSize > 10 * 1024 * 1024) { // > 10MB
                    System.out.println("Medium file detected (" + fileSize / 1024 / 1024 + "MB). Using streaming approach.");
                    corpus = readFileWithStreaming(filePath);
                } else {
                    System.out.println("Small file detected. Loading entirely into memory.");
                    corpus = Files.readString(Paths.get(filePath));
                }
                System.out.println("Successfully processed " + corpus.length() + " characters from file.");
            } catch (IOException e) {
                System.err.println("Error reading file: " + filePath);
                System.err.println("Error details: " + e.getMessage());
                System.err.println("Using default corpus instead.");
                corpus = getDefaultCorpus();
            }
        } else {
            System.out.println("No file specified. Usage: java Main <path-to-text-file>");
            System.out.println("Using default corpus for demonstration.");
            corpus = getDefaultCorpus();
        }
        
        corpus = corpus.toLowerCase();  // normalize case
        // Train BPE tokenizer on the corpus
        int vocabSize = 280;  // target vocabulary size (includes base 256 bytes) - small number of merges for effective tokenization
        BPETokenizer tokenizer = new BPETokenizer(vocabSize);
        tokenizer.train(corpus);
        // Encode corpus text to token sequence
        List<Integer> tokenList = tokenizer.encode(corpus);
        int N = tokenList.size();
        
        // Check if we have enough tokens for training
        if (N < 2) {
            System.out.println("Error: Need at least 2 tokens for training, but got " + N);
            return;
        }
        
        System.out.println("Successfully tokenized corpus into " + N + " tokens");
        
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
        double learningRate = 0.001;  // reduced learning rate for stability
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
                    double expVal = Math.exp(logits[t][v] - maxLogit);
                    sumExp += expVal;
                }
                // Add small epsilon to prevent log(0)
                sumExp = Math.max(sumExp, 1e-15);
                
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
    
    /**
     * Returns a default corpus for demonstration when no file is provided.
     */
    private static String getDefaultCorpus() {
        return "the cat in the hat sits on the mat. " +
               "the dog runs in the park with the ball. " +
               "a bird flies high in the blue sky above the trees. " +
               "children play games in the playground after school. " +
               "the sun shines bright on a beautiful summer day. " +
               "flowers bloom in the garden during spring season. " +
               "students read books in the quiet library room. " +
               "the ocean waves crash against the rocky shore. " +
               "mountains stand tall covered with white snow. " +
               "farmers work hard in the fields growing crops. " +
               "the moon glows softly in the dark night sky. " +
               "rain falls gently on the green grass below. " +
               "cars drive slowly through the busy city streets. " +
               "birds sing sweet songs from the tree branches. " +
               "the cat sleeps peacefully on the warm windowsill. " +
               "children laugh and play in the sunny backyard. " +
               "the river flows quietly through the valley. " +
               "snow falls silently on the winter landscape. " +
               "people walk together in the crowded market. " +
               "the wind blows leaves across the autumn ground. " +
               "fish swim freely in the clear blue water. " +
               "clouds move slowly across the vast open sky. " +
               "the fire burns warmly in the old stone fireplace. " +
               "stars twinkle brightly in the midnight darkness. " +
               "the cat chases the mouse around the house. " +
               "dogs bark loudly when strangers approach the gate. " +
               "flowers smell sweet in the morning garden air. " +
               "the train travels fast along the steel railroad tracks. " +
               "children build sandcastles on the sandy beach shore. " +
               "the lighthouse guides ships safely through the night.";
    }
}