import * as tf from "@tensorflow/tfjs";
import { getPhishingDataAsTensors } from "./get-phishing-data.js";
import { setUIToReadyState } from "./ui.js";
import { defineModel, hyperParameters } from "./define-model.js";
import { interpretPredictionProbsAtThreshold, plotROCCurve } from "./plot-roc-curve.js";

const TRAIN_BUTTON = document.getElementById("train-button");

let dataTensors;
let model;
const trainLogs = [];

async function prepareForModelUsage() {
  dataTensors = await getPhishingDataAsTensors();
  model = defineModel(30);

  setUIToReadyState();
}

async function trainModel(model) {
  const { trainFeatures, trainTarget, testFeatures, testTarget } = dataTensors;
  await model.fit(trainFeatures, trainTarget, {
    ...hyperParameters,
    callbacks: {
      onEpochBegin: (epoch) => {
        if ((epoch + 1) % 100 === 0 || epoch === 0 || epoch === 2 || epoch === 4) {
          const predictionsAtEpoch = model.predict(testFeatures);
          predictionsAtEpoch.print();
          const auc = plotROCCurve(epoch, predictionsAtEpoch, testTarget);
          console.log(auc);
        }
      },
      onEpochEnd: (epoch, logs) => {
        trainLogs.push(logs);
      },
    },
  });
}

function evaluateModel(model) {
  const { testFeatures, testTarget } = dataTensors;
  return tf.tidy(() => {
    const evalResults = model.evaluate(testFeatures, testTarget, { batchSize: hyperParameters.batchSize });
    const testDataPredictions = model.predict(testFeatures);
    const interpretedPredictions = interpretPredictionProbsAtThreshold(testDataPredictions);

    const evalLoss = evalResults[0].dataSync();
    const evalAccuracy = evalResults[1].dataSync();
    const recall = tf.metrics.recall(testTarget, interpretedPredictions).dataSync()[0];
    const precision = tf.metrics.precision(testTarget, interpretedPredictions).dataSync()[0];

    return { evalLoss, evalAccuracy, recall, precision };
  });
}

TRAIN_BUTTON.addEventListener("click", async () => {
  await trainModel(model);
  console.log("EVALUATING MODEL");
  const { recall, precision, evalLoss, evalAccuracy } = evaluateModel(model);
  const finalTestingLog = trainLogs[trainLogs.length - 1];

  document.getElementById("ending-loss").innerText = `Final Training Loss ${finalTestingLog.loss}`;
  document.getElementById("ending-val-loss").innerText = `Final Training Val Loss ${finalTestingLog.val_loss}`;
  document.getElementById("eval-loss").innerText = `Eval Loss ${evalLoss}`;
  document.getElementById("eval-accuracy").innerText = `Eval Accuracy ${evalAccuracy}`;
  document.getElementById("eval-recall").innerText = `Eval Recall ${recall}`;
  document.getElementById("eval-precision").innerText = `Eval Precision ${precision}`;
});

(() => {
  prepareForModelUsage();
})();
