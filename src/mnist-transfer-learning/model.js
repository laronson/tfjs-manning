import * as tf from "@tensorflow/tfjs";

const BASE_MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json";

export async function loadBaseModel() {
  const baseModel = await tf.loadLayersModel(BASE_MODEL_URL);
  return baseModel;
}
