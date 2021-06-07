import pprint

import cv2

import numpy as np

from openvino.inference_engine import IECore

from visualisation_utils import visualise


class Model:
    def __init__(self, name, precision, device):
        ie = IECore()

        XML_PATH = (
            f"./{name}_saved_model/openvino/{precision}/movenet_singlepose_{name}_3.xml"
        )
        BIN_PATH = (
            f"./{name}_saved_model/openvino/{precision}/movenet_singlepose_{name}_3.bin"
        )
        if name == "thunder":
            self.input_size = (256, 256)
        elif name == "lightning":
            self.input_size = (192, 192)
        else:
            print("Unsupported Model")

        net = self.ie.read_network(model=XML_PATH, weights=BIN_PATH)

        self.input_blob = next(iter(net.input_info))

        self.exec_net = ie.load_network(net, device_name=device, num_requests=1)
        self.inference_request = self.exec_net.requests[0]

    def infer(self, image_path):
        image = cv2.imread(image_path)
        img = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]

        self.exec_net.infer(inputs={self.input_blob: img})
        pprint.pprint(self.inference_request.output_blobs)
        outputs = self.inference_request.output_blobs["Identity"].buffer

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
    model = Model(name="lightning", precision="FP16", device="CPU")
    model.infer("test.png")
