import * as tf from "@tensorflow/tfjs";

export function defineConvnetModel(imgHeight, imgWidth) {
  const model = tf.sequential();

  //Input shape is in the form of HWC.  Since we are dealing with greyscale images we only need to worry about one
  //color channel so C=1.  Using a kernel size of 3 means that we will perform dot product operations on the input of
  //the model using a 3x3 pixel filter.  In turn,
  model.add(tf.layers.conv2d({ inputShape: [imgHeight, imgWidth, 1], kernelSize: 3, filters: 16, activation: "relu" }));
  model.add(tf.layers.maxPool2d({ poolSize: 2, strides: 2 }));

  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" }));

  //At this point, the images that are being analyzed by the model have gone through the convnet portion of the model
  //It is now time for the model to switch gears to using dense layers to analyze the output of the convnet and make
  //a prediction based off of the output of the convnet
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));

  //In the dense layer portion of our model, we pass the outpuit of the convnet layers to a dense layer using the softmax
  //activation function which will (for the case of analyzing the MNIST dataset) have 10 units each representing the
  //probability of which number 0-9 the image the model is analyzing represents.
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  compileConvnetModel(model);

  return model;
}

function compileConvnetModel(model) {
  //We are going to use the rmsprop (root mean squared propagation) optimizer in our model which is an popular choice
  //in training convnets. The optimizer has adaptive learning rates so we do not need to handle that.  It does this by
  //keeping track of the moving average of the squared gradients observed during training and adjusts the learning rate
  //accordingly.  It is particularly good at handling sparse gradients (aka gradients produced by sparsely connected
  //layers) and using those gradients to find relevant features.
  const optimizer = "rmsprop";
  const lossFn = "categoricalCrossentropy";
  const metrics = ["accuracy"];

  model.compile({ optimizer, loss: lossFn, metrics });
}
