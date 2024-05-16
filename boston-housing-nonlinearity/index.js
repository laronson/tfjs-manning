import { loadHousingData, featureDescriptions } from "./housingData.js";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

const TRAIN_1HIDDEN_BUTTON = document.getElementById("nn-mlr-1hidden");
const TRAIN_2HIDDEN_BUTTON = document.getElementById("nn-mlr-2hidden");

let tensors = {};
let baselineStatus = {};

const LEARNING_RATE = 0.01;

function modelBaselineEvaluationValues(trainTarget, testTarget) {
  const baselineTarget = trainTarget.mean();
  const baselineLoss = testTarget.sub(baselineTarget).square().mean();
  return {
    baselineTarget: baselineTarget.dataSync()[0],
    baselineLoss: baselineLoss.dataSync()[0],
  };
}

function normalizeDataSet(featureTensor2d) {
  const trainFeaturesMean = featureTensor2d.mean(0);
  const diffFromMean = featureTensor2d.sub(trainFeaturesMean);
  const std = diffFromMean.square().mean(0).sqrt();
  return diffFromMean.div(std);
}

function multiLayerPerceptronRegression1HiddenLayer(numFeatures) {
  let model = tf.sequential();

  /**
   * In this layer, we use the sigmoid activation function.  This function is the concatination of two line segments
   * (between zero and one) and the line that represents the activation function is smooth and differentiable at every
   * point on the line.  This makes it possible to perform backpropigation on the model during training
   */
  model.add(
    tf.layers.dense({
      inputShape: [numFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );

  //This layer still uses the default linear activation function
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });
  model.summary();

  return model;
}

function multiLayerPerceptronRegression2HiddenLayer(numFeatures) {
  let model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [numFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(
    tf.layers.dense({
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });
  model.summary();

  return model;
}

async function trainModel(model, modelName, trainFeatures, trainTarget) {
  const trainLogs = [];
  const normalizedTrainingFeatures = normalizeDataSet(trainFeatures);
  const container = document.querySelector(`#${modelName} .chart`);

  await model.fit(normalizedTrainingFeatures, trainTarget, {
    validationSplit: 0.2,
    epochs: 200,
    batchSize: 20,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch: ${epoch}, loss: ${logs.loss}, val_loss: ${logs.val_loss}`
        );
        trainLogs.push(logs);
        tfvis.show.history(container, trainLogs, ["loss", "val_loss"]);
      },
    },
  });
}

function evaluateModel(model, testFeatures, testTarget) {
  const normalizedTestingFeatures = normalizeDataSet(testFeatures);
  tf.util.shuffleCombo(normalizedTestingFeatures, testTarget);
  const testLoss = model
    .evaluate(normalizedTestingFeatures, testTarget, {
      batchSize: 40,
    })
    .dataSync()[0];
  return testLoss;
}

async function run(model, modelName) {
  const { trainFeatures, trainTarget, testFeatures, testTarget } = tensors;
  const { baselineLoss, baselineTarget } = baselineStatus;

  const trainingResults = await trainModel(
    model,
    modelName,
    trainFeatures,
    trainTarget
  );
  console.log("after train");

  const testLoss = evaluateModel(model, testFeatures, testTarget);
  console.log("after test");

  console.log(`baseline Target: ${baselineTarget}`);
  console.log(`baseline Loss: ${baselineLoss}`);
  console.log(`Training Results: ${trainingResults}`);
  console.log(`Testing Loss ${testLoss}`);
}

TRAIN_1HIDDEN_BUTTON.addEventListener("click", async () => {
  const model = multiLayerPerceptronRegression1HiddenLayer(
    featureDescriptions.length
  );
  await run(model, "oneHidden");
});

TRAIN_2HIDDEN_BUTTON.addEventListener("click", async () => {
  const model = multiLayerPerceptronRegression2HiddenLayer(
    featureDescriptions.length
  );
  await run(model, "twoHidden");
});

async function prepareForModelUsage() {
  tensors = await loadHousingData();
  baselineStatus = modelBaselineEvaluationValues(
    tensors.trainTarget,
    tensors.testTarget
  );

  setUIToReadyState();
}

function setUIToReadyState() {
  document.getElementById("status").innerText = "Data Loaded";
  document.getElementById("baselineStatus").innerText =
    "Baseline Status Computed";

  TRAIN_1HIDDEN_BUTTON.removeAttribute("disabled");
  TRAIN_2HIDDEN_BUTTON.removeAttribute("disabled");
}
(async () => {
  await prepareForModelUsage();
})();
