import * as tf from "@tensorflow/tfjs";
import { addROCCurve } from "./ui";

const thresholds = [
  0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94,
  0.96, 0.98, 1.0,
];

export function plotROCCurve(epoch, predictionProbsAtEpoch, testTarget) {
  return tf.tidy(() => {
    let auc = 0;
    const tprs = [];
    const fprs = [];

    for (let i = 0; i < thresholds.length; i++) {
      const threshold = thresholds[i];
      console.log({ threshold });
      const predictionValues = interpretPredictionProbsAtThreshold(predictionProbsAtEpoch, threshold);
      predictionValues.print();
      const { tpr, fpr } = getROCMetrics(predictionValues, testTarget);
      tprs.push(tpr);
      fprs.push(fpr);

      if (threshold !== 0) {
        auc += ((tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i])) / 2;
      }
    }
    addROCCurve(fprs, tprs, epoch);
    return auc;
  });
}

export function interpretPredictionProbsAtThreshold(predictionProbs, threshold) {
  if (threshold == null) {
    threshold = 0.5;
  }

  //Ensure that the threshold values cannot be any greater than 100% or less than 0%
  tf.util.assert(threshold >= 0 && threshold <= 1, `Expected threshold to be >=0 and <=1, but got ${threshold}`);

  //Returns a tensors of boolean values that is the same shape as predictions.  At each index of the returned value, the
  //boolean will represent if value at that index of the predictions array is greater than the threshold
  //Need to use tf.scalar here to create a rank-0 tensor with the desired threshold for the comparison
  const predictionValueCondition = predictionProbs.greater(tf.scalar(threshold));

  return tf.where(predictionValueCondition, tf.onesLike(predictionProbs), tf.zerosLike(predictionProbs));
}

function getROCMetrics(predictions, trueValues) {
  const zero = tf.scalar(0);
  const one = tf.scalar(1);

  const truePositives = tf.logicalAnd(predictions.equal(one), trueValues.equal(one)).sum().cast("float32");
  const trueNegatives = tf.logicalAnd(predictions.equal(zero), trueValues.equal(zero)).sum().cast("float32");

  const fpr = truePositives.div(truePositives.add(trueNegatives)).dataSync()[0];
  const tpr = tf.metrics.recall(trueValues, predictions).dataSync()[0];
  console.log({ tpr, fpr });

  return { tpr, fpr };
}
