import * as tf from "@tensorflow/tfjs";

import { loadAndPrepareData } from "./data.js";
import { loadBaseModel } from "./model.js";
import { trainModel } from "./train.js";
import {
  enableRetrainFunctionality,
  getActionButtons,
  getEpochCount,
  getTrainingMode,
  setSimplePrediction,
} from "./ui.js";

let model;
let data;
let trainedModel;

const { loadHostedModelBtn, retrainBtn, simplePredictBtn } = getActionButtons();

async function prepareModelForTraining() {
  model = await loadBaseModel();
  enableRetrainFunctionality();
}

function simplePredict(model) {
  tf.tidy(() => {
    const randomExampleIdx = Math.floor(Math.random() * data.testData.imageTensor.shape[0]);

    const predictTensor = data.testData.imageTensor.slice([randomExampleIdx], [1]);
    const labelTensor = data.testData.labelTensor.slice([randomExampleIdx], [1]);

    const prediction = model.predict(predictTensor);
    setSimplePrediction(
      `Prediction: ${parseInt(prediction.argMax(1).dataSync()) + 5}, Actual Label: ${
        parseInt(labelTensor.argMax(1).dataSync()) + 5
      }`
    );
  });
}

retrainBtn.addEventListener("click", async () => {
  const trainingMode = getTrainingMode();
  const epochCount = getEpochCount();
  trainedModel = await trainModel(model, trainingMode, data, epochCount);
});

simplePredictBtn.addEventListener("click", () => {
  simplePredict(trainedModel);
});

loadHostedModelBtn.addEventListener("click", async () => {
  await prepareModelForTraining();
  data = await loadAndPrepareData();
});
