import os
import cv2
import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
from torchvision import transforms
from collections import namedtuple, OrderedDict
from utils import parse_class_index_file, letterbox

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class RecognitionBackend(nn.Module):
    def __init__(self, weights, device):
        super().__init__()
        self.device = device
        # Create logger
        logger = trt.Logger(trt.Logger.INFO)
        # Create Binding
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.bindings = OrderedDict()
        # Deserialize engine.
        with open(weights, "rb") as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())

        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            shape = tuple(model.get_binding_shape(i))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        # Create context
        self.context = model.create_execution_context()
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings["input_0"].shape[0]

    def forward(self, img):
        assert img.shape == self.bindings['input_0'].shape, (img.shape, self.bindings['input_0'].shape)
        self.binding_addrs['input_0'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output_0'].data

        return y

    def warmup(self, imgsz=(1, 3, 224, 224), half=True):
        im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
        self.forward(im)


if __name__ == '__main__':
    img_path = "images/demo.jpg"
    engine_weight_path = "logs/best_model.engine"
    device = torch.device('cuda:0')
    convNeXt = RecognitionBackend(engine_weight_path, device)
    # convNeXt.warmup()
    # Inference
    my_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load image, [3, H, W] --> [1, 3, 224, 224]
    img = cv2.imread(img_path)
    # img = letterbox(img, [224, 224], stride=64, auto=False)[0]
    img = cv2.resize(img, (224, 224))
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device)
    # im = im.half() / 255
    im = im.half()
    # expand for batch dim
    if len(im.shape) == 3:
        im = im[None]

    # parse class_index.
    class_index = parse_class_index_file('class_index.json')

    predict_tensor = convNeXt(im)
    print(predict_tensor)
    predict_tensor = torch.squeeze(predict_tensor).cpu()
    print(predict_tensor)
    predict = torch.softmax(predict_tensor, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    res = "class: {}, prob: {:.3}".format(class_index[str(predict_cla)], predict[predict_cla].numpy())
    print(res)