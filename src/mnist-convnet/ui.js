const MODEL_TRAINED_TEXT = "Model has been trained."; //hacky way to check if model has already been trained for model select

export function getTrainButton() {
  return document.getElementById("train-model-button");
}

export function getPredictOneButton() {
  return document.getElementById("predict-one");
}

export function getModelType() {
  return document.getElementById("model-select").value;
}

export function getModelPath() {
  return document.getElementById("model-path-input").value;
}

export function setStatusToReady() {
  document.getElementById("status").innerText = "training data ready";
  getTrainButton().removeAttribute("disabled");
}

export function setStatusToTrained() {
  document.getElementById("status").innerText = MODEL_TRAINED_TEXT;
  getPredictOneButton().removeAttribute("disabled");
}

export function getTrainingEpochs() {
  return Number.parseInt(document.getElementById("train-epochs").value);
}

export function logTrainingStatus(batchPercentage) {
  document.getElementById(
    "training-status"
  ).innerText = `Training... (${batchPercentage}% complete). To stop training, refresh or close page.`;
}

export function drawPrediction(image, target, prediction) {
  const height = 28;
  const width = 28;
  const imageDiv = document.getElementById("prediction-images");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");

  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  const predText = document.createElement("p");
  predText.innerText = `Prediction ${prediction}`;
  const targetText = document.createElement("p");
  targetText.innerText = `Target ${target}`;

  imageDiv.appendChild(canvas);
  imageDiv.appendChild(predText);
  imageDiv.appendChild(targetText);
}

const modelSelect = document.getElementById("model-select");
modelSelect.addEventListener("change", () => {
  if (modelSelect.value === "browser") {
    if (document.getElementById("status").innerText !== MODEL_TRAINED_TEXT) {
      getPredictOneButton().setAttribute("disabled", true);
    }
  } else {
    getPredictOneButton().removeAttribute("disabled");
  }
});
