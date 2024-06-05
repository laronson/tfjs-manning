import * as tf from "@tensorflow/tfjs-node";
import { ArgumentParser } from "argparse";
import { model } from "./model.js";
import { loadData } from "./data.js";

function getRunParameters() {
  const parser = new ArgumentParser({
    description: "TensorFlow.js-Node MNIST Example.",
    add_help: true,
  });

  parser.add_argument("--epochs", {
    type: "int",
    default: 20,
    help: "Number of epochs to use when training the model",
  });

  parser.add_argument("--batch_size", {
    type: "int",
    default: 128,
    help: "Batch size to use when training the model",
  });

  parser.add_argument("--should_save_model", {
    type: "boolean",
    help: "should the model be saved to file",
  });

  return parser.parse_args();
}

async function trainModel({ epochs, batchSize, trainImages, trainLabels }) {
  tf.util.assert(
    trainImages.length === trainLabels.length,
    `Mismatch in the number of images (${trainImages.length}) and ` + `the number of labels (${trainLabels.length})`
  );

  const imagesTensor = tf.reshape(trainImages, [trainImages.length, 28, 28, 1]);
  const labelsTensor = tf.oneHot(tf.squeeze(trainLabels), 10);

  const trainResults = await model.fit(imagesTensor, labelsTensor, {
    epochs,
    batchSize,
    validationSplit: 0.15,
  });
  return trainResults;
}

function evaluateModel({ testImages, testLabels }) {
  tf.tidy(() => {
    const imagesTensor = tf.reshape(testImages, [testImages.length, 28, 28, 1]);
    const labelsTensor = tf.oneHot(tf.squeeze(testLabels), 10);

    const evaluationResults = model.evaluate(imagesTensor, labelsTensor);
    console.log(
      `\nEvaluation result:\n` +
        `  Loss = ${evaluationResults[0].dataSync()[0].toFixed(3)}; ` +
        `Accuracy = ${evaluationResults[1].dataSync()[0].toFixed(3)}`
    );
  });
}

async function run({ epochs, batchSize, shouldSaveModel }) {
  console.log(
    `Starting with parameters Epochs: ${epochs}, batchSize: ${batchSize}, shouldSaveModel: ${shouldSaveModel}`
  );
  const { trainImages, trainLabels, testImages, testLabels } = await loadData();
  console.log("got data");
  await trainModel({ epochs, batchSize, trainImages: trainImages, trainLabels: trainLabels });
  evaluateModel({ testImages, testLabels });

  if (shouldSaveModel) {
    const modelSavePath = "/Users/leonard/Desktop/manning-tfjs-projects/src/mnist-convnet/server/.saved-model";
    await model.save(`file://${modelSavePath}`);
    console.log(`Model saved to: ${modelSavePath}`);
  }
}

const args = getRunParameters();
run({ epochs: args.epochs, batchSize: args.batch_size, shouldSaveModel: args.should_save_model });
