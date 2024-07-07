import os
import json
import torch
from PIL import Image
from torchvision import transforms
from convNeXt import ConvNeXt
from utils import parse_class_index_file

if __name__ == '__main__':
    num_classes = 5
    input_shape = 224  # Model Input.
    model_path = "logs/best_model.pth"
    img_path = "images/demo.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(int(input_shape * 1.14)),
         transforms.CenterCrop(input_shape),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # parse class_index.
    class_index = parse_class_index_file('class_index.json')

    # create model
    model = ConvNeXt(num_classes=5, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
    # load model weights
    model_weight = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weight)
    model.eval()

    # inference class.
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # Returns the index of the maximum value in a dimension.

    res = "class: {}, prob: {:.3}".format(class_index[str(predict_cla)], predict[predict_cla].numpy())
    print(res)