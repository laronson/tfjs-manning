import * as tf from "@tensorflow/tfjs";

const MNIST_5_TO_9_DATA_URLS = {
  train: "https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/gte5.train.json",
  test: "https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/gte5.test.json",
};

export async function loadAndPrepareData() {
  const [rawTrainData, rawTestData] = await loadData();
  const trainData = convertDataToTensors(rawTrainData, 5);
  const testData = convertDataToTensors(rawTestData, 5);

  return { trainData, testData, raw: { rawTrainData, rawTestData } };
}

async function loadData() {
  const [trainResponse, testResponse] = await Promise.all([
    fetch(MNIST_5_TO_9_DATA_URLS.train),
    fetch(MNIST_5_TO_9_DATA_URLS.test),
  ]);

  return Promise.all([trainResponse.json(), testResponse.json()]);
}

function convertDataToTensors(rawData, outputValuesCount) {
  const numExamples = rawData.length;
  const rowDataLength = rawData[0].x.length;
  const colDataLength = rawData[0].x[0].length;

  const { imgData, labelData } = rawData.reduce(
    (exampleData, example) => {
      exampleData.imgData.push(example.x);
      exampleData.labelData.push(example.y);
      return exampleData;
    },
    { imgData: [], labelData: [] }
  );

  let imageTensor = tf.reshape(tf.tensor3d(imgData, [numExamples, rowDataLength, colDataLength]), [
    numExamples,
    rowDataLength,
    colDataLength,
    1,
  ]);

  imageTensor = imageTensor.div(tf.scalar(255));

  const labelTensor = tf.oneHot(tf.tensor1d(labelData, "int32"), outputValuesCount);

  return { imageTensor, labelTensor };
}
