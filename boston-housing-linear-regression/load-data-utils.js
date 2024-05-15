const Papa = require("papaparse");
const BASE_URL =
  "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/";

/**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
export const loadCsv = async (filename) => {
  return new Promise((resolve) => {
    const url = `${BASE_URL}${filename}`;

    console.log(`  * Downloading data from: ${url}`);
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results["data"]));
      },
    });
  });
};

/**
 * Given CSV data returns an array of arrays of numbers.
 *
 * @param {Array<Object>} data Downloaded data.
 *
 * @returns {Promise.Array<number[]>} Resolves to data with values parsed as floats.
 */
const parseCsv = async (data) => {
  return new Promise((resolve) => {
    data = data.map((row) => {
      return Object.keys(row).map((key) => parseFloat(row[key]));
    });
    resolve(data);
  });
};
