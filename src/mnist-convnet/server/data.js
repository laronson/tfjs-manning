import { createWriteStream, existsSync, readFile } from "fs";
import { get as httpsGet } from "https";
import { join, dirname } from "path";
import { createGunzip } from "zlib";
import { promisify } from "util";
import { fileURLToPath } from "url";
import assert from "assert";

const BASE_DATA_RETRIEVAL_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const BASE_FILE_LOCATION = "./.data/";
const TRAIN_FEATURE_FILENAME = "train-images-idx3-ubyte";
const TRAIN_TARGET_FILENAME = "train-labels-idx1-ubyte";
const TEST_FEATURE_FILENAME = "t10k-images-idx3-ubyte";
const TEST_TARGET_FILENAME = "t10k-labels-idx1-ubyte";
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const readFileAsync = promisify(readFile);

export async function loadData() {
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    loadImages(TRAIN_FEATURE_FILENAME),
    loadLabels(TRAIN_TARGET_FILENAME),
    loadImages(TEST_FEATURE_FILENAME),
    loadLabels(TEST_TARGET_FILENAME),
  ]);

  return {
    trainImages,
    trainLabels,
    testImages,
    testLabels,
  };
}

async function fetchAndSaveFile(filename) {
  const url = `${BASE_DATA_RETRIEVAL_URL}${filename}.gz`; //Could prob use js URL object here for safety
  const fileLoc = join(dirname(fileURLToPath(import.meta.url)), BASE_FILE_LOCATION, filename);

  if (existsSync(fileLoc)) {
    return readFileAsync(fileLoc);
  }

  const file = createWriteStream(fileLoc);
  const writeToFile = new Promise((resolve) => {
    httpsGet(url, (response) => {
      const unzip = createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on("end", () => {
        console.log(`unzipped ${fileLoc}`);
        resolve(readFileAsync(fileLoc));
      });
    });
  });
  return writeToFile;
}

async function loadImages(filename) {
  console.log(`Loading Images for ${filename}`);
  const imageDataBuffer = await fetchAndSaveFile(filename);
  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(imageDataBuffer, headerBytes);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

  const images = [];
  let idx = headerBytes;

  while (idx < imageDataBuffer.byteLength) {
    const imageData = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      imageData[i] = imageDataBuffer.readUInt8(idx++) / 255;
    }
    images.push(imageData);
  }

  assert.equal(images.length, headerValues[1]);
  return images;
}

async function loadLabels(filename) {
  console.log(`Loading labels for ${filename}`);
  const labelsDataBuffer = await fetchAndSaveFile(filename);
  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(labelsDataBuffer, headerBytes);
  assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

  const labels = [];
  let idx = headerBytes;
  while (idx < labelsDataBuffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = labelsDataBuffer.readUInt8(idx++);
    }
    labels.push(array);
  }

  assert.equal(labels.length, headerValues[1]);
  return labels;
}

function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka big-endian)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}
