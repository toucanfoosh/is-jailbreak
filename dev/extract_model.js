const fs = require("fs");
const path = require("path");

// 0) Check if dist/models exists
if (!fs.existsSync("../dist")) {
    fs.mkdirSync("../dist");
}
if (!fs.existsSync("../dist/models")) {
    fs.mkdirSync("../dist/models");
}

// 1) Load your model.json
const modelJson = JSON.parse(fs.readFileSync("./models/js_graph_model/model.json", "utf-8"));

// Extract modelTopology
const modelTopology = modelJson.modelTopology;

// Flatten the "weights" from the weightsManifest entries
const weightSpecs = [];
if (modelJson.weightsManifest && Array.isArray(modelJson.weightsManifest)) {
    for (const manifest of modelJson.weightsManifest) {
        if (manifest.weights) {
            weightSpecs.push(...manifest.weights);
        }
    }
}

// Write them to separate JSON files for convenience
// fs.writeFileSync("../dist/models/modelTopology.json", JSON.stringify(modelTopology));
// fs.writeFileSync("../dist/models/weightSpecs.json", JSON.stringify(weightSpecs));

// console.log("Extracted modelTopology.json and weightSpecs.json");

// Suppose your model.json listed the bin files in this order:
const binFiles = ["models/js_graph_model/group1-shard1of2.bin", "models/js_graph_model/group1-shard2of2.bin"];

// 1) Read each bin file as a Buffer
let buffers = binFiles.map((f) => fs.readFileSync(`./${f}`));
// 2) Concatenate
let totalLength = buffers.reduce((acc, b) => acc + b.length, 0);
let combined = Buffer.concat(buffers, totalLength);

// 3) Base64-encode
let base64Str = combined.toString("base64");

// 4) Save to a JSON file
// fs.writeFileSync(
//     "../dist/models/weightDataBase64.json",
//     JSON.stringify(base64Str)
// );

// console.log("Created weightDataBase64.json");

// Copy tfidf_vocab over to src
fs.copyFileSync("./models/tfidf_vocab.json", "../dist/models/tfidf_vocab.json");

// Copy js_graph_model dir to src
const srcDir = "../dist/models";
if (!fs.existsSync(srcDir)) {
    fs.mkdirSync(srcDir);
}
fs.readdirSync("./models/js_graph_model").forEach((f) => {
    fs.copyFileSync(`./models/js_graph_model/${f}`, `${srcDir}/${f}`);
});

console.log("Extraction Complete")
