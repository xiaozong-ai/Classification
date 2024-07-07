import os
import torch
import onnxruntime
from PIL import Image
import numpy as np
from torchvision import transforms
from utils import parse_class_index_file


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    img_path = "images/demo.jpg"
    model_path = 'logs/best_model.onnx'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    my_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load image, [3, H, W] --> [1, 3, 224, 224]
    image = Image.open("images/demo.jpg").convert("RGB")
    image = my_transform(image)             # [3, H, W] --> [3, 224, 224]
    image = torch.unsqueeze(image, dim=0)   # [3, 224, 224] --> [1, 3, 224, 224]
    image = to_numpy(image)

    # parse class_index.
    class_index = parse_class_index_file('class_index.json')

    # Build inferenceSession object.
    inference_session = onnxruntime.InferenceSession("logs/best_model.onnx")

    # Compute ONNX Runtime output prediction
    output = inference_session.run(None, {'input_0': image})
    output = np.squeeze(output[0])
    predict_tensor = torch.tensor(output)
    predict = torch.softmax(predict_tensor, dim=0)
    predict_cla = torch.argmax(predict).numpy()

    res = "class: {}, prob: {:.3}".format(class_index[str(predict_cla)], predict[predict_cla].numpy())
    print(res)