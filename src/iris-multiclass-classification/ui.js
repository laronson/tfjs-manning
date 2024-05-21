import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { IRIS_CLASSES } from "./iris-data.js";

export function alertUIModelAndDataAreReadyForTraining() {
  const DATA_STATUS = document.getElementById("status");
  const TRAIN_BUTTON = document.getElementById("train-button");
  DATA_STATUS.innerText = "Data Ready";
  TRAIN_BUTTON.removeAttribute("disabled");
}

export function drawTrainingGraphs(trainLogs) {
  const lossGraph = document.getElementById("loss-graph");
  const accGraph = document.getElementById("acc-graph");

  tfvis.show.history(lossGraph, trainLogs, ["loss", "val_loss"]);
  tfvis.show.history(accGraph, trainLogs, ["acc", "val_acc"]);
}

export async function drawConfusionMatrix(predictionTensor, truthValuesTensor) {
  const confusionMatrix = document.getElementById("confusion-matrix");

  const { predictions, labels } = tf.tidy(() => {
    const predictions = predictionTensor.argMax(-1);
    const labels = truthValuesTensor.argMax(-1);
    return { predictions, labels };
  });

  const confusionMatrixMetrics = await tfvis.metrics.confusionMatrix(labels, predictions);
  tfvis.render.confusionMatrix(
    confusionMatrix,
    { values: confusionMatrixMetrics, labels: IRIS_CLASSES },
    { shadeDiagonal: true }
  );

  tf.dispose([predictions, labels]);
}

export function getTrainButton() {
  return document.getElementById("train-button");
}
