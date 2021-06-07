import pprint
import time

import cv2

import numpy as np

import tensorflow.lite as tflite

from visualisation_utils import visualise


class Model:
    def __init__(self, name):
        self.model_path = f"./{name}_saved_model/model_float32.tflite"

        if name == "thunder":
            self.input_size = (256, 256)
        elif name == "lightning":
            self.input_size = (192, 192)
        else:
            print("Unsupported Model")

        self.interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=4)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, image_path):
        image = cv2.imread(image_path)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.input_size)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype("float32")

        self.interpreter.set_tensor(self.input_details[0]["index"], frame)
        start_time = time.time()
        self.interpreter.invoke()
        stop_time = time.time()
        print("time: ", stop_time - start_time)

        outputs = self.interpreter.get_tensor(self.output_details[0]["index"])
        outputs = np.squeeze(np.array(outputs))
        pprint.pprint(outputs.shape)
        pprint.pprint(outputs)

        outputImage = visualise(
            frame=image.copy(),
            coords=outputs[:, [1, 0]] * [image.shape[1], image.shape[0]],
            scores=outputs[:, 2],
            score_thresh=0.1,
        )
        cv2.imshow("output", outputImage)
        cv2.waitKey(0)


if __name__ == "__main__":
    model = Model(name="lightning")
    model.infer("test.png")
