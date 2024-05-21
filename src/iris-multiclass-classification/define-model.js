import * as tf from "@tensorflow/tfjs";
import { NUM_FEATURES } from "./iris-data.js";

export const hyperParameters = {
  epochs: 40,
};

export function defineModel() {
  return tf.tidy(() => {
    const model = tf.sequential();

    model.add(tf.layers.dense({ inputShape: [NUM_FEATURES], units: 10, activation: "sigmoid" }));
    model.add(tf.layers.dense({ activation: "softmax", units: 3 }));

    const optimizer = tf.train.adam(0.01);
    model.compile({
      optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    model.summary();

    return model;
  });
}
