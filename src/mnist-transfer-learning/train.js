import * as tf from "@tensorflow/tfjs";
import * as tfVis from "@tensorflow/tfjs-vis";

import { getProgressBarCallbackConfig } from "./ui.js";

const trainModeStrings = {
  freezeFeatureLayers: "freeze-feature-layers",
  noFreezing: "no-freezing",
  reinitializeWeights: "reinitialize-weights",
};

export async function trainModel(model, trainMode, data, epochCount) {
  const { trainData, testData } = data;
  const trainingModel = await prepareBaseModel(model, trainMode);

  trainingModel.compile({
    loss: "categoricalCrossentropy",
    //Originally had the learningRate for the optimizer set to .01 but that was too high for my computer and was throwing
    //off my results.  Had to change the learning rate to be higher or just let TF do the work of setting the appropriate
    //rate for me in this case
    optimizer: tf.train.adam(0.01),
    metrics: ["accuracy"],
  });
  trainingModel.summary();

  await trainingModel.fit(trainData.imageTensor, trainData.labelTensor, {
    batchSize: 128,
    shuffle: true,
    validationSplit: 0.15,
    epochs: epochCount,
    validationData: [testData.imageTensor, testData.labelTensor],
    callbacks: [
      getProgressBarCallbackConfig(),
      tfVis.show.fitCallbacks({ name: trainMode, tab: "Transfer Learning" }, ["val_loss", "val_acc"], {
        zoomToFit: true,
        zoomToFitAccuracy: true,
        height: 200,
        callbacks: ["onEpochEnd"],
      }),
    ],
  });

  return trainingModel;
}

//There is a bug here!!!
//If you choose to freeze then reinitilize you will end up using the same model and your results will be influenced
//by this
async function prepareBaseModel(model, trainMode) {
  console.log(trainMode);
  if (trainMode === trainModeStrings.freezeFeatureLayers) {
    return setFreezeFeatureLayers(model, false);
  } else if (trainMode === trainModeStrings.reinitializeWeights) {
    const resetFreezeModel = setFreezeFeatureLayers(model, true);
    return reinitializeWeights(resetFreezeModel);
  }
  return setFreezeFeatureLayers(model, true);
}

function setFreezeFeatureLayers(model, isTrainable) {
  for (let i = 0; i < 7; i++) {
    model.layers[i].trainable = isTrainable;
  }
  return model; //Is this returning a reference? Probably
}

async function reinitializeWeights(model) {
  const returnString = false;
  const reInitializedModel = await tf.models.modelFromJSON({
    modelTopology: model.toJSON(null, returnString),
  });
  return reInitializedModel;
}
