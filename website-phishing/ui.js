import * as tfvis from "@tensorflow/tfjs-vis";

const DATA_LOADING_STATUS = document.getElementById("status");
const TRAIN_BUTTON = document.getElementById("train-button");

const rocSeries = [];
const rocValues = [];

export function setUIToReadyState() {
  DATA_LOADING_STATUS.innerText = "Data Loaded";
  TRAIN_BUTTON.removeAttribute("disabled");
}

export function addROCCurve(fprs, tprs, epoch) {
  const seriesName = `epoch #${epoch + 1}`; //Make sure epochs are not zero based when displayed on chart
  rocSeries.push(seriesName);

  const newRocSeriesValues = [];
  for (let i = 0; i < fprs.length; i++) {
    newRocSeriesValues.push({
      x: fprs[i],
      y: tprs[i],
    });
  }
  rocValues.push(newRocSeriesValues);

  return tfvis.render.scatterplot(
    document.getElementById("rocCurve"),
    { values: rocValues, series: rocSeries },
    {
      width: 450,
      height: 320,
      xLabel: "FPR",
      yLabel: "TPR",
    }
  );
}
