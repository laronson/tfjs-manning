import { loadCsv } from "./load-data-utils.js";
import * as tf from "@tensorflow/tfjs";

export const featureDescriptions = [
  "Crime rate",
  "Land zone size",
  "Industrial proportion",
  "Next to river",
  "Nitric oxide concentration",
  "Number of rooms per house",
  "Age of housing",
  "Distance to commute",
  "Distance to highway",
  "Tax rate",
  "School class size",
  "School drop-out rate",
];

const TRAIN_FEATURES_FN = "train-data.csv";
const TRAIN_TARGET_FN = "train-target.csv";
const TEST_FEATURES_FN = "test-data.csv";
const TEST_TARGET_FN = "test-target.csv";

export async function loadHousingData() {
  const [trainFeaturesArr, trainTargetArr, testFeaturesArr, testTargetArr] =
    await Promise.all([
      loadCsv(TRAIN_FEATURES_FN),
      loadCsv(TRAIN_TARGET_FN),
      loadCsv(TEST_FEATURES_FN),
      loadCsv(TEST_TARGET_FN),
    ]);

  tf.util.shuffleCombo(trainFeaturesArr, trainTargetArr);
  tf.util.shuffleCombo(testFeaturesArr, testTargetArr);

  return {
    trainFeatures: tf.tensor2d(trainFeaturesArr),
    trainTarget: tf.tensor2d(trainTargetArr),
    testFeatures: tf.tensor2d(testFeaturesArr),
    testTarget: tf.tensor2d(testTargetArr),
  };
}
