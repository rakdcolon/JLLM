import java.util.*;
import java.nio.charset.StandardCharsets;

/** Byte Pair Encoding (BPE) Tokenizer implementation. */
class BPETokenizer {
    /** Mapping from token ID to the byte sequence it represents. For base tokens 0-255, the sequence is the single byte. */
    private Map<Integer, byte[]> tokenToBytes;
    /** List of merge rules learned (in order). Each rule merges a pair of tokens into a new token. */
    private List<MergeRule> merges;
    /** Desired vocabulary size after BPE merges (including base tokens). */
    private int vocabSize;
    /** A simple struct representing a merge rule (two tokens merging into one). */
    private static class MergeRule {
        int first; 
        int second; 
        int newToken;
        MergeRule(int a, int b, int newToken) {
            this.first = a;
            this.second = b;
            this.newToken = newToken;
        }
    }

    /** Constructor specifying target vocabulary size (including base tokens). */
    public BPETokenizer(int vocabSize) {
        this.vocabSize = vocabSize;
        this.tokenToBytes = new HashMap<>();
        this.merges = new ArrayList<>();
        // Initialize base vocabulary (0-255 cover all byte values)
        for (int i = 0; i < 256; i++) {
            // Each byte value maps to itself as a single-byte sequence
            this.tokenToBytes.put(i, new byte[]{(byte) i});
        }
    }

    /** Train BPE on the given corpus text (learn merge rules). */
    public void train(String corpus) {
        // Get initial token sequence from text (byte-level tokenization)
        byte[] bytes = corpus.getBytes(StandardCharsets.UTF_8);
        List<Integer> tokens = new ArrayList<>();
        for (byte b : bytes) {
            // convert to unsigned int 0-255
            tokens.add(b & 0xFF);
        }
        // We will perform merges until vocab size is reached or no more pairs.
        // The base vocabulary size is 256 (all possible byte values).
        int baseVocabSize = 256;
        int nextTokenId = baseVocabSize;
        // We want total vocab = vocabSize, so number of merges = vocabSize - baseVocabSize
        int mergesToDo = this.vocabSize - baseVocabSize;
        if (mergesToDo < 0) {
            mergesToDo = 0;
        }
        for (int m = 0; m < mergesToDo; m++) {
            // Count frequency of each adjacent token pair in current token list
            Map<Long, Integer> pairFreq = new HashMap<>();
            // Encode a pair of tokens (a,b) into one long key for counting
            for (int i = 0; i < tokens.size() - 1; i++) {
                int a = tokens.get(i);
                int b = tokens.get(i+1);
                long pairKey = (((long) a) << 32) | (b & 0xffffffffL);
                pairFreq.put(pairKey, pairFreq.getOrDefault(pairKey, 0) + 1);
            }
            if (pairFreq.isEmpty()) {
                break; // no pairs to merge (e.g., very short text)
            }
            // Find the most frequent pair
            long bestPairKey = 0;
            int bestCount = -1;
            for (Map.Entry<Long, Integer> entry : pairFreq.entrySet()) {
                int count = entry.getValue();
                if (count > bestCount) {
                    bestCount = count;
                    bestPairKey = entry.getKey();
                }
            }
            if (bestCount <= 0) {
                break; // no pair occurs more than once
            }
            // Decode pairKey into token IDs
            int tokenA = (int) (bestPairKey >> 32);
            int tokenB = (int) bestPairKey;
            // Create new token representing the merged pair
            int newToken = nextTokenId++;
            // Define the byte sequence for the new token as concatenation of the two tokens' sequences
            byte[] seqA = tokenToBytes.get(tokenA);
            byte[] seqB = tokenToBytes.get(tokenB);
            byte[] mergedSeq = new byte[seqA.length + seqB.length];
            System.arraycopy(seqA, 0, mergedSeq, 0, seqA.length);
            System.arraycopy(seqB, 0, mergedSeq, seqA.length, seqB.length);
            tokenToBytes.put(newToken, mergedSeq);
            // Record this merge rule
            merges.add(new MergeRule(tokenA, tokenB, newToken));
            // Apply this merge to the token list: replace every occurrence of [tokenA, tokenB] with [newToken]
            List<Integer> newTokens = new ArrayList<>();
            for (int i = 0; i < tokens.size();) {
                if (i < tokens.size() - 1 && tokens.get(i) == tokenA && tokens.get(i+1) == tokenB) {
                    newTokens.add(newToken);
                    i += 2; // skip the merged pair
                } else {
                    newTokens.add(tokens.get(i));
                    i += 1;
                }
            }
            tokens = newTokens;
        }
        // After training, 'merges' holds the merge rules and 'tokenToBytes' has the vocabulary mappings.
    }

    /** Encode a new text using the learned BPE merges. Returns a list of token IDs. */
    public List<Integer> encode(String text) {
        // Start with byte-level tokens
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        List<Integer> tokens = new ArrayList<>();
        for (byte b : bytes) {
            tokens.add(b & 0xFF);
        }
        // Apply each merge rule in order
        for (MergeRule rule : merges) {
            List<Integer> newTokens = new ArrayList<>();
            for (int i = 0; i < tokens.size();) {
                if (i < tokens.size() - 1 && tokens.get(i) == rule.first && tokens.get(i+1) == rule.second) {
                    newTokens.add(rule.newToken);
                    i += 2;
                } else {
                    newTokens.add(tokens.get(i));
                    i += 1;
                }
            }
            tokens = newTokens;
        }
        return tokens;
    }

    /** Decode a sequence of token IDs back to the original text. */
    public String decode(List<Integer> tokens) {
        // Reconstruct byte sequence from tokens
        int totalLength = 0;
        for (int token : tokens) {
            byte[] seq = tokenToBytes.get(token);
            totalLength += seq.length;
        }
        byte[] bytes = new byte[totalLength];
        int pos = 0;
        for (int token : tokens) {
            byte[] seq = tokenToBytes.get(token);
            System.arraycopy(seq, 0, bytes, pos, seq.length);
            pos += seq.length;
        }
        // Decode bytes to UTF-8 string
        return new String(bytes, StandardCharsets.UTF_8);
    }
}