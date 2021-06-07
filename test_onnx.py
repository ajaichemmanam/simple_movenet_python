import onnx
import onnxruntime
import numpy as np
import cv2
import pprint

from visualisation_utils import visualise


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
        print(image.shape)
        frame = cv2.resize(image, self.input_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype("float32")

        inputs = {self.sess.get_inputs()[0].name: frame}
        outputs = self.sess.run(None, inputs)

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
    model = Model(name="thunder")
    model.infer("test.png")
