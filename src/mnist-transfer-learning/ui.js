import * as tf from "@tensorflow/tfjs";

const LOAD_HOSTED_MODEL_BUTTON = document.getElementById("load-model-and-data-button");
const RETRAIN_BUTTON = document.getElementById("retrain-button");
const STATUS_TEXT = document.getElementById("status");
const TRAINING_MODE_SELECT = document.getElementById("training-mode-select");
const EPOCH_INPUT = document.getElementById("epochs-input");
const SIMPLE_PREDICT_TEST = document.getElementById("simple-Predict-text");
const SIMPLE_PREDICT_BUTTON = document.getElementById("simple-predict-button");

export function getActionButtons() {
  return {
    loadHostedModelBtn: LOAD_HOSTED_MODEL_BUTTON,
    retrainBtn: RETRAIN_BUTTON,
    simplePredictBtn: SIMPLE_PREDICT_BUTTON,
  };
}

export function enableRetrainFunctionality() {
  LOAD_HOSTED_MODEL_BUTTON.setAttribute("disabled", true);
  setStatus("Model and Data Loaded.  Ready for Training");
  RETRAIN_BUTTON.removeAttribute("disabled");
}

export function disableRetrainFunctionality() {
  RETRAIN_BUTTON.setAttribute("disabled", true);
}

export function getTrainingMode() {
  return TRAINING_MODE_SELECT.value;
}

export function getEpochCount() {
  return EPOCH_INPUT.value;
}

export function setSimplePrediction(text) {
  SIMPLE_PREDICT_TEST.innerText = text;
}

function setStatus(text, color = "black") {
  STATUS_TEXT.textContent = text;
  STATUS_TEXT.style.color = color;
}

export function getProgressBarCallbackConfig() {
  const epochs = getEpochCount();
  // Custom callback for updating the progress bar at the end of epochs.

  const trainProg = document.getElementById("trainProg");
  let beginMillis;
  const progressBarCallbackConfig = {
    onTrainBegin: async (logs) => {
      beginMillis = tf.util.now();
      setStatus("Please wait and do NOT click anything while the model retrains...", "blue");
      trainProg.value = 0;
    },
    onTrainEnd: async (logs) => {
      console.log(logs);

      setStatus(
        `Done retraining ${epochs} epochs (elapsed: ` +
          `${(tf.util.now() - beginMillis).toFixed(1)} ms` +
          `). Standing by.`,
        "black"
      );
    },
    onEpochEnd: async (epoch, logs) => {
      console.log(logs);

      setStatus(
        `Please wait and do NOT click anything while the model ` + `retrains... (Epoch ${epoch + 1} of ${epochs})`
      );
      trainProg.value = ((epoch + 1) / epochs) * 100;
    },
  };
  return progressBarCallbackConfig;
}
