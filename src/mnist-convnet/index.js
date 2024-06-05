import * as tf from "@tensorflow/tfjs";

import { convertImageBufferTo4DTensor, convertLabelBufferTo2DTensor, getData } from "./data.js";
import { getModel } from "./define-convnet-model.js";
import {
  setStatusToReady,
  getTrainingEpochs,
  logTrainingStatus,
  getTrainButton,
  getPredictOneButton,
  setStatusToTrained,
  drawPrediction,
  getModelType,
} from "./ui.js";

let model;
let imageData;
const imgHeight = 28;
const imgWidth = 28;

async function trainModel() {
  const { trainImages, trainLabels } = imageData;
  const batchSize = 320;
  const validationSplit = 0.15;
  const epochs = getTrainingEpochs();
  let trainBatchCount = 0;

  const imageTensor = convertImageBufferTo4DTensor(trainImages);
  const labelsTensor = convertLabelBufferTo2DTensor(trainLabels);

  const numberOfBatches = Math.ceil((imageTensor.shape[0] * (1 - validationSplit)) / batchSize) * epochs;

  await model.fit(imageTensor, labelsTensor, {
    batchSize,
    validationSplit,
    epochs,
    callbacks: {
      onBatchEnd: () => {
        trainBatchCount++;
        logTrainingStatus(((trainBatchCount / numberOfBatches) * 100).toFixed(1));
      },
      onEpochEnd: (epoch, logs) => {
        console.log({ epoch, acc: logs.val_acc, loss: logs.val_loss });
      },
    },
  });

  imageTensor.dispose();
  labelsTensor.dispose();
}

function evaluateModel() {
  tf.tidy(() => {
    const { testImages, testLabels } = imageData;
    const imageTensor = convertImageBufferTo4DTensor(testImages);
    const labelsTensor = convertLabelBufferTo2DTensor(testLabels);

    const results = model.evaluate(imageTensor, labelsTensor);
    const testAccPercent = results[1].dataSync()[0] * 100;
    console.log(`Evaluation phase accuracy was ${testAccPercent.toFixed(1)}`);
  });
}

function predictOne() {
  tf.tidy(() => {
    const { testImages, testLabels } = imageData;
    const imageTensor = convertImageBufferTo4DTensor(testImages);
    const labelsTensor = convertLabelBufferTo2DTensor(testLabels);

    const randomIdx = Math.floor(Math.random() * imageTensor.shape[0]);
    const img = imageTensor.slice([randomIdx], 1);
    const label = labelsTensor.slice([randomIdx], 1);

    const prediction = model.predict(img);

    drawPrediction(img, label.argMax(1).dataSync(), prediction.argMax(1).dataSync());
  });
}

async function run() {
  imageData = await getData();
  setStatusToReady();

  const trainButton = getTrainButton();
  const predictOneButton = getPredictOneButton();

  trainButton.addEventListener("click", async () => {
    model = await getModel({ modelType: getModelType(), imgHeight, imgWidth });
    document.getElementById("model-select").setAttribute("disabled", true);
    await trainModel();
    evaluateModel();
    setStatusToTrained();
  });
  predictOneButton.addEventListener("click", async () => {
    if (!model) {
      model = await getModel({ modelType: getModelType(), imgHeight, imgWidth });
    }
    predictOne();
  });
}

run();
