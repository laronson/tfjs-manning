import { sequential, layers } from "@tensorflow/tfjs-node";

const model = sequential();

//For the model that is being trained using node instead of in the browser, we can add another conv2d layer before
//passing to a maxpooling layer to run more filters on the data because we have more compute power.
model.add(layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: "relu" }));
model.add(layers.conv2d({ filters: 32, kernelSize: 3, activation: "relu" }));
model.add(layers.maxPooling2d({ poolSize: 2 }));

model.add(layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
model.add(layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
model.add(layers.maxPooling2d({ poolSize: 2 }));

model.add(layers.flatten());

//Add a dropout layer to the model before passing to the dense layers so we can better prevent overfitting using the
//dropout strategy
model.add(layers.dropout({ rate: 0.25 }));
model.add(layers.dense({ units: 512, activation: "relu" }));

model.add(layers.dropout({ rate: 0.5 }));
model.add(layers.dense({ units: 10, activation: "softmax" }));

model.compile({
  optimizer: "rmsprop",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

model.summary();

export { model };
