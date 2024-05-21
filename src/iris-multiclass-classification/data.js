import * as tf from "@tensorflow/tfjs";
import { IRIS_DATA, IRIS_NUM_CLASSES, NUM_FEATURES } from "./iris-data.js";

export function prepareIrisData() {
  return tf.tidy(() => {
    const irisDataCopy = [...IRIS_DATA];
    tf.util.shuffle(irisDataCopy);

    const { trainFeaturesArr, trainTargetsArr, testFeaturesArr, testTargetsArr } =
      splitToTrainAndTestArrays(irisDataCopy);

    const trainFeatures = tf.tensor2d(trainFeaturesArr);
    //Must call toInt on the 1d tensor here because tf.oneHot() expects the input tensor to be of type int32.  If we
    //do not do this conversion, the contents of the 1d tensor will be of type float32
    const trainTargets = tf.oneHot(tf.tensor1d(trainTargetsArr).toInt(), IRIS_NUM_CLASSES);

    const testFeatures = tf.tensor2d(testFeaturesArr);
    const testTargets = tf.oneHot(tf.tensor1d(testTargetsArr).toInt(), IRIS_NUM_CLASSES);

    return { trainFeatures, trainTargets, testFeatures, testTargets };
  });
}
function splitToTrainAndTestArrays(data) {
  const trainDataSet = [];
  const testDataSet = [];
  const splitData = splitDataSetByTarget(data);

  for (const split of splitData) {
    const trainDataLength = Math.ceil(split.length * 0.8);
    trainDataSet.push(...split.slice(0, trainDataLength));
    testDataSet.push(...split.slice(trainDataLength));
  }
  tf.util.shuffle(trainDataSet);

  const { featuresArr: trainFeaturesArr, targetsArr: trainTargetsArr } = getIrisFeatureAndTargetArrays(trainDataSet);
  const { featuresArr: testFeaturesArr, targetsArr: testTargetsArr } = getIrisFeatureAndTargetArrays(testDataSet);

  return { trainFeaturesArr, trainTargetsArr, testFeaturesArr, testTargetsArr };
}

function splitDataSetByTarget(data) {
  const splitData = data.reduce(
    (prev, irisData) => {
      prev[irisData[NUM_FEATURES]].push(irisData);
      return prev;
    },
    { 0: [], 1: [], 2: [] }
  );

  return [splitData[0], splitData[1], splitData[2]];
}

function getIrisFeatureAndTargetArrays(irisDataSet) {
  const featuresArr = [];
  const targetsArr = [];

  for (const irisData of irisDataSet) {
    featuresArr.push(irisData.slice(0, NUM_FEATURES));
    targetsArr.push(irisData[irisData.length - 1]);
  }
  return { featuresArr, targetsArr };
}
