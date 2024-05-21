import * as tf from "@tensorflow/tfjs";
import { loadCsv } from "../util/load-csv.js";

const BASE_URL =
  "https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/";

const TRAIN_DATA = "train-data.csv";
const TRAIN_TARGET = "train-target.csv";
const TEST_DATA = "test-data.csv";
const TEST_TARGET = "test-target.csv";

export async function getRawPhishingData() {
  const [trainFeaturesArr, trainTargetArr, testFeaturesArr, testTargetArr] = await Promise.all([
    loadCsv(BASE_URL, TRAIN_DATA),
    loadCsv(BASE_URL, TRAIN_TARGET),
    loadCsv(BASE_URL, TEST_DATA),
    loadCsv(BASE_URL, TEST_TARGET),
  ]);

  return { trainFeaturesArr, trainTargetArr, testFeaturesArr, testTargetArr };
}

export async function getPhishingDataAsTensors() {
  const { trainFeaturesArr, trainTargetArr, testFeaturesArr, testTargetArr } = await getRawPhishingData();

  console.log({ trainFeaturesArr, trainTargetArr, testFeaturesArr, testTargetArr });

  //Should not need to shuffle data here because there is no inherent ordering to this data so there is not a risk of
  //the model picking up on a unintended pattern from a specific data ordering.
  return {
    trainFeatures: tf.tensor2d(trainFeaturesArr),
    trainTarget: tf.tensor2d(trainTargetArr),
    testFeatures: tf.tensor2d(testFeaturesArr),
    testTarget: tf.tensor2d(testTargetArr),
  };
}
