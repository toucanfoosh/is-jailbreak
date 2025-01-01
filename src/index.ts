import * as tf from "@tensorflow/tfjs";
import * as tfn from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";

let model: tf.GraphModel | null = null;
let tfidfMeta: any | null = null;

// Resolve paths relative to the current module
const modelsDir = path.join(__dirname, "../models");
const tfidfMetaPath = path.join(modelsDir, "tfidf_vocab.json");
const modelPath = path.join(modelsDir, "model.json");

/**
 * Loads the TF.js model and prepares it for use.
 */
export async function loadJailbreak(): Promise<void> {
  if (!model) {
    try {
      // Load TF-IDF metadata
      if (!tfidfMeta) {
        const metaData = fs.readFileSync(tfidfMetaPath, "utf-8");
        tfidfMeta = JSON.parse(metaData);
        console.log("TF-IDF metadata loaded successfully");
      }

      // Load the model
      console.log("Loading model from:", modelPath);
      const handler = tfn.io.fileSystem(modelPath);
      model = await tf.loadGraphModel(handler);
      console.log("Jailbreak model loaded successfully");
    } catch (error) {
      console.error("Error loading the model or metadata:", error);
      throw new Error("Failed to load model or metadata.");
    }
  }
}

/**
 * Preprocesses input text using the TF-IDF vocabulary and configuration.
 * @param input The input string to preprocess.
 * @returns A tensor representation of the input.
 */
function preprocessInput(input: string): tf.Tensor {
  if (!tfidfMeta) {
    throw new Error("TF-IDF metadata not loaded. Call loadJailbreak() first.");
  }

  const { vocabulary, ngram_range, max_features, idf } = tfidfMeta;
  const tokens = input.toLowerCase().split(/\W+/);
  const featureVector = new Array(max_features).fill(0);

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    for (let n = ngram_range[0]; n <= ngram_range[1]; n++) {
      if (i + n - 1 < tokens.length) {
        const ngram = tokens.slice(i, i + n).join(" ");
        if (vocabulary[ngram] !== undefined) {
          featureVector[vocabulary[ngram]] += 1;
        }
      }
    }
  }

  for (let i = 0; i < featureVector.length; i++) {
    featureVector[i] *= idf[i] || 0;
  }

  return tf.tensor2d([featureVector], [1, max_features]);
}

/**
 * Determines if the given input string is a jailbreak prompt.
 * @param input The input string to analyze.
 * @returns A promise resolving to a boolean indicating jailbreak detection.
 */
export async function isJailbreak(input: string): Promise<boolean> {
  if (!model) {
    throw new Error("Model not loaded. Call loadJailbreak() first.");
  }

  const inputTensor = preprocessInput(input);
  const prediction = model.predict(inputTensor) as tf.Tensor;
  const probabilities = await prediction.softmax().data();

  inputTensor.dispose();
  prediction.dispose();

  // Assume index 1 corresponds to "jailbreak"
  return probabilities[1] > 0.5;
}
