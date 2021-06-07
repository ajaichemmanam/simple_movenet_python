import pprint
import time

import cv2

import numpy as np

import tensorflow.lite as tflite


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
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = np.expand_dims(image, axis=0)
        image = image.astype("float32")

        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        start_time = time.time()
        self.interpreter.invoke()
        stop_time = time.time()
        print("time: ", stop_time - start_time)

        scores = self.interpreter.get_tensor(self.output_details[0]["index"])
        pprint.pprint(scores)


if __name__ == "__main__":
    model = Model(name="lightning")
    model.infer("test.png")
