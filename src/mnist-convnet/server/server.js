import express from "express";
import cors from "cors";

const PORT = 8080;
let app = express();

app.use(cors());

app.use(
  "/model.json",
  express.static(`/Users/leonard/Desktop/manning-tfjs-projects/src/mnist-convnet/server/.saved-model/model.json`)
);

app.use(
  "/weights.bin",
  express.static(`/Users/leonard/Desktop/manning-tfjs-projects/src/mnist-convnet/server/.saved-model/weights.bin`)
);

app.listen(PORT, function () {
  console.log(`Starting server on port ${PORT}`);
});
