import { loadHousingData, featureDescriptions } from "./housingData.js";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

const TRAIN_BUTTON = document.getElementById("simple-mlr");
const CHART_CONTAINER = document.getElementById("tfvis-chart");

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

function generateAndCompileModel(numFeatures) {
  let model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [numFeatures], units: 1 }));
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });
  model.summary();

  return model;
}

async function trainModel(model, trainFeatures, trainTarget) {
  const trainLogs = [];
  const normalizedTrainingFeatures = normalizeDataSet(trainFeatures);

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
        tfvis.show.history(CHART_CONTAINER, trainLogs, ["loss", "val_loss"]);
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

async function run() {
  const { trainFeatures, trainTarget, testFeatures, testTarget } =
    await loadHousingData();
  const { baselineLoss, baselineTarget } = modelBaselineEvaluationValues(
    trainTarget,
    testTarget
  );

  const model = generateAndCompileModel(featureDescriptions.length);

  TRAIN_BUTTON.addEventListener("click", async () => {
    const trainingResults = await trainModel(model, trainFeatures, trainTarget);

    const testLoss = evaluateModel(model, testFeatures, testTarget);
    console.log(`baseline Target: ${baselineTarget}`);
    console.log(`baseline Loss: ${baselineLoss}`);
    console.log(`Training Results: ${trainingResults}`);
    console.log(`Testing Loss ${testLoss}`);
  });
}
(async () => {
  await run();
})();

// Final train-set loss: 21.9864
// Final validation-set loss: 31.1396
// Test-set loss: 25.3206
// Baseline loss: 85.58
