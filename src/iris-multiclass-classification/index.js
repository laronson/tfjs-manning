import * as tf from "@tensorflow/tfjs";
import { prepareIrisData } from "./data.js";
import { defineModel, hyperParameters } from "./define-model.js";
import {
  alertUIModelAndDataAreReadyForTraining,
  drawConfusionMatrix,
  drawTrainingGraphs,
  getTrainButton,
} from "./ui.js";

const trainLogs = [];

async function trainModel(model, trainFeatures, trainTargets, testFeatures, testTargets) {
  await model.fit(trainFeatures, trainTargets, {
    //Need to dig into why using test data as validation data here
    validationData: [testFeatures, testTargets],
    epochs: hyperParameters.epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        tf.tidy(() => {
          console.log(`epoch ${epoch}`);
          trainLogs.push(logs);
          drawTrainingGraphs(trainLogs);

          const trainingPredictions = model.predict(testFeatures);
          drawConfusionMatrix(trainingPredictions, testTargets);
        });
      },
    },
  });
}

function prepareForModelUsage() {
  const { trainFeatures, trainTargets, testFeatures, testTargets } = prepareIrisData();
  const model = defineModel();
  const trainButton = getTrainButton();

  trainButton.addEventListener("click", async () => {
    await trainModel(model, trainFeatures, trainTargets, testFeatures, testTargets);
  });

  alertUIModelAndDataAreReadyForTraining();
}

prepareForModelUsage();
