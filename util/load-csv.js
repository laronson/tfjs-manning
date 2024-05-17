const Papa = require("papaparse");
import { parseCsv } from "./parse-csv.js";

/**
 * Downloads and returns the csv.
 *
 * @param {string} filename Name of file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
export const loadCsv = async (baseUrl, filename) => {
  return new Promise((resolve) => {
    const url = `${baseUrl}${filename}`;

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
