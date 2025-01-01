import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";

// Type definitions for the TF-IDF metadata
interface TfidfMetadata {
  vocabulary: { [key: string]: number };
  idf: number[];
  ngram_range: [number, number];
  max_features: number;
  stop_words: string | null;
  lowercase: boolean;
}

let model: tf.GraphModel | null = null;
let tfidfMetadata: TfidfMetadata | null = null;

/**
 * Loads the pre-trained TensorFlow.js model from the specified path.
 */
async function loadModel(): Promise<void> {
  if (!model) {
    try {
      console.log("Loading model from path:", `file://${__dirname}/model.json`);
      model = await tf.loadGraphModel(`file://${__dirname}/model.json`);
      console.log("Model loaded successfully.");
    } catch (error) {
      console.error("Error loading the model:", error);
    }
  } else {
    console.log("Model is already loaded.");
  }
}

/**
 * Loads the TF-IDF metadata from a JSON file.
 */
function loadTfidfMetadata(): void {
  try {
    const metadataRaw = fs.readFileSync(
      `${__dirname}/tfidf_vocab.json`,
      "utf-8"
    );
    tfidfMetadata = JSON.parse(metadataRaw) as TfidfMetadata;
    console.log("TF-IDF metadata loaded successfully.");
  } catch (error) {
    console.error("Error loading TF-IDF metadata:", error);
  }
}

/**
 * Loads the pre-trained model and TF-IDF metadata.
 */
export async function loadJailbreak(): Promise<void> {
  await loadModel();
  loadTfidfMetadata();
}

/**
 * Preprocesses a string prompt into a tensor for classification.
 *
 * @param prompt - The input string prompt.
 * @returns A tensor suitable for model prediction.
 */
function preprocessPrompt(prompt: string): tf.Tensor {
  if (!tfidfMetadata) {
    throw new Error(
      "TF-IDF metadata not loaded. Call loadTfidfMetadata() first."
    );
  }

  // Lowercase the input text
  let processedText = prompt;
  if (tfidfMetadata.lowercase) {
    processedText = processedText.toLowerCase();
  }

  // Generate n-grams
  const tokens = processedText.split(/\s+/); // Tokenize by whitespace
  const ngrams = generateNgrams(tokens, tfidfMetadata.ngram_range);

  // Map n-grams to vocabulary indices
  const featureVector = new Array(tfidfMetadata.max_features).fill(0);
  for (const ngram of ngrams) {
    const index = tfidfMetadata.vocabulary[ngram];
    if (index !== undefined && index < tfidfMetadata.max_features) {
      featureVector[index] += 1; // Term frequency (TF)
    }
  }

  // Apply IDF weights
  const transformedVector = featureVector.map(
    (tf, index) => tf * tfidfMetadata!.idf[index]
  );

  // Convert to a 2D tensor (required by the model)
  return tf.tensor2d([transformedVector], [1, tfidfMetadata.max_features]);
}

/**
 * Generates n-grams from tokens based on a specified range.
 *
 * @param tokens - The array of tokens (words).
 * @param ngramRange - The range of n-grams to generate (e.g., [1, 3]).
 * @returns An array of n-grams.
 */
function generateNgrams(
  tokens: string[],
  ngramRange: [number, number]
): string[] {
  const [minN, maxN] = ngramRange;
  const ngrams: string[] = [];

  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i <= tokens.length - n; i++) {
      ngrams.push(tokens.slice(i, i + n).join(" "));
    }
  }

  return ngrams;
}

/**
 * Classifies a given prompt using the loaded model.
 *
 * @param input - Preprocessed input data.
 * @returns Model prediction.
 */
export async function classify(input: tf.Tensor): Promise<tf.Tensor | null> {
  if (!model) {
    console.error("Model not loaded. Call loadModel() first.");
    return null;
  }

  try {
    console.log("Preprocessed Input Tensor:", input.arraySync());
    const prediction = model.predict(input) as tf.Tensor;
    return prediction;
  } catch (error) {
    console.error("Error during prediction:", error);
    return null;
  }
}

/**
 * Determines if a given prompt is classified as a jailbreak.
 *
 * @param prompt - The string prompt to evaluate.
 * @param threshold - A confidence threshold between 0 and 1 (inclusive).
 * @returns A boolean indicating whether the prompt exceeds the threshold.
 */
export async function isJailbreak(
  prompt: string,
  threshold: number = 0.5
): Promise<boolean> {
  if (threshold < 0 || threshold > 1) {
    throw new Error("Threshold must be between 0 and 1 (inclusive).");
  }

  console.log(`Evaluating prompt: "${prompt}" with threshold: ${threshold}`);

  if (!model) {
    console.error("Model not loaded. Call loadModel() first.");
    return false;
  }

  if (!tfidfMetadata) {
    console.error(
      "TF-IDF metadata not loaded. Call loadTfidfMetadata() first."
    );
    return false;
  }

  try {
    // Preprocess the prompt
    const preprocessedInput = preprocessPrompt(prompt);

    // Get the classification result
    const prediction = await classify(preprocessedInput);

    if (prediction) {
      // Extract the confidence score for the "jailbreak" class
      const probabilities = prediction.arraySync() as number[][];
      console.log("Probabilities:", probabilities);
      const jailbreakConfidence = probabilities[0][1]; // Adjust index if needed

      console.log(
        `Jailbreak confidence: ${jailbreakConfidence}, Decision: ${
          jailbreakConfidence > threshold
        }`
      );
      return jailbreakConfidence > threshold;
    }

    return false;
  } catch (error) {
    console.error("Error in isJailbreak:", error);
    return false;
  }
}
