import * as tf from "@tensorflow/tfjs";

export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 30000;

const NUM_TRAIN_ELEMENTS = 25000;
const STORAGE_SIZE_PER_IMAGE = 4;

const MNIST_IMAGES_SPRITE_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS_PATH = "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

export async function getData() {
  const img = new Image();
  const chunkSize = 5000;

  img.crossOrigin = ""; //we must set the crossOrigin value to an empty string to avoid error

  const getImageDataRequest = new Promise((resolve) => {
    img.onload = () => {
      //set the hight and width of the images from the height and width of the image tag to the height and width of the
      //image that is loaded.  In this case, that image is the sprite we load with all of the number images
      img.width = img.naturalWidth;
      img.height = img.naturalHeight;

      const imageDataFloat32Array = parseSpriteChunk(img, chunkSize);
      resolve(imageDataFloat32Array);
    };
    //In order for the onLoad function to run, we need to set the img.src value for the image
    img.src = MNIST_IMAGES_SPRITE_PATH;
  });
  //Stupidly letting then chain happen cuz trying to make code patterns match for how we get image data
  const getImageLabelsRequest = fetch(MNIST_LABELS_PATH).then((labelData) => {
    return labelData.arrayBuffer().then((labelArrayBuffer) => {
      return new Uint8Array(labelArrayBuffer);
    });
  });

  const [imageData, imageLabels] = await Promise.all([getImageDataRequest, getImageLabelsRequest]);

  const trainImages = imageData.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
  const testImages = imageData.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS, NUM_DATASET_ELEMENTS * IMAGE_SIZE);
  const trainLabels = imageLabels.slice(0, NUM_TRAIN_ELEMENTS * NUM_CLASSES);
  const testLabels = imageLabels.slice(NUM_TRAIN_ELEMENTS * NUM_CLASSES, NUM_DATASET_ELEMENTS * NUM_CLASSES);

  return { trainImages, testImages, trainLabels, testLabels };
}

function parseSpriteChunk(image, chunkSize) {
  //Create a array buffer with the length of the array set to hold all image provided in the MNIST sprite image
  // We multiply this number by 4 because we will eventually use this byte array to create a Float32Array for which
  //each picture is stored in for bytes worth of data.  Therefore, the size is set to:
  //numOfElements * PxPerImage * bytesOfStoragePerImage
  const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * STORAGE_SIZE_PER_IMAGE);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = image.naturalWidth;
  canvas.height = chunkSize;

  for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
    //Create a new Float32Array for each iteration so we can interact with a small portion of the datasetBytesBuffer
    //using floating point integers.  We need to do this conversion because we do our normalization step here by dividing
    //each greyscale color channel value by 255
    const imageDataArray = new Float32Array(
      datasetBytesBuffer,
      i * chunkSize * IMAGE_SIZE * STORAGE_SIZE_PER_IMAGE,
      IMAGE_SIZE * chunkSize
    );

    //Draw the portion of the sprite image we are trying to parse on the canvas and then get the pixel data of the
    //image that we drew on the canvas
    ctx.drawImage(image, 0, i * chunkSize, image.width, chunkSize, 0, 0, image.width, chunkSize);
    const imageData = ctx.getImageData(0, 0, image.width, chunkSize);

    //Get the image data from the pixel data we just obtained and store it on the imageDataArray which in turn will
    //propagate those changes to the datasetBytesBuffer that we passed in
    for (let j = 0; j < imageData.data.length / 4; j++) {
      //In its current state, the HWC values of each image is 28x28x3 however, because this is a greyscale image, all
      //values on every color channel are the same,  therefore, we only need to store one (in this case red) of the
      //color channel values
      imageDataArray[j] = imageData.data[j * 4] / 255; //dividing by 255 to normalize each px cuz it is a greyscale image
    }
  }
  //Return the entire image data set as a Float32Array
  return new Float32Array(datasetBytesBuffer);
}

export function convertImageBufferTo4DTensor(imageBuffer) {
  //Images stored in typed arrays (in this case Float32Array types) are in a flat structure.  By providing both the image
  //buffer and the desired shape of our new 4D tensor, we tell tensorflow how we want the imageBuffer to be broken
  //up when creating the new tensor and tensorflow will do it for us.
  return tf.tensor4d(imageBuffer, [imageBuffer.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
}

export function convertLabelBufferTo2DTensor(labelBuffer) {
  //Labels are also stored in flat structures but we want the data to be in the format of arrays of size 10 consisting
  //of 0s and 1s representing which number is displayed in respective image
  return tf.tensor2d(labelBuffer, [labelBuffer.length / NUM_CLASSES, NUM_CLASSES]);
}
