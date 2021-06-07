import onnx
import onnxruntime
import numpy as np
import cv2
import pprint


class Model:
    def __init__(self, name):
        if name == "thunder":
            model_path = "./thunder_saved_model/model_float32.onnx"
            self.input_size = (256, 256)
        elif name == "lightning":
            model_path = "./lightning_saved_model/model_float32.onnx"
            self.input_size = (192, 192)
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)

        self.sess = onnxruntime.InferenceSession(model_path)

    def infer(self, image_path):
        image = cv2.imread(image_path)
        frame = cv2.resize(image, self.input_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype("float32")

        inputs = {self.sess.get_inputs()[0].name: frame}
        outputs = self.sess.run(None, inputs)

        pprint.pprint(outputs)


if __name__ == "__main__":
    model = Model(name="thunder")
    model.infer("test.png")
