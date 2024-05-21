import * as tf from "@tensorflow/tfjs";

export const hyperParameters = {
  epochs: 400,
  batchSize: 350,
  validationSplit: 0.2,
};

export function defineModel(numFeatures) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [numFeatures],
      units: 100,
      activation: "sigmoid",
    })
  );
  model.add(tf.layers.dense({ units: 100, activation: "sigmoid" }));

  /**
   * Because this is a binary classification problem, the output layer of this model uses a sigmoid activation
   * function and will only return values between 0 and 1.  This is in place of the default linear activation that can
   * return any number (reliably within the model's capacity) which is good for more linear examples like the boston
   * housing example.
   * This also enables us to use the value between 1 and 0 to act as a level of confidence that the output belongs
   * to the positive case.  We can also use this value during training as an easy way to calculate a value for our loss
   * function by using the output's distance away from the actual answer.
   */
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  /**
   * Different than the sgd optimizer where you have to set a learning threshold before training, the adam optimizer
   * intelligently changes the learning rate throughout training to best find weights that result in minimum loss.  As
   * a result, the adam optimizer leads to better convergence and less dependence on the choice of learning rate to
   * avoid the scenario where a too high of a learning rate is chosen leading to "zig-zags" and too low or a rate is
   * chosen leading to a slow and less optimal training process.
   */
  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
