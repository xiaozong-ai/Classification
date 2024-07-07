import torch
import torch.nn

from convNeXt import ConvNeXt

model_path = "logs/best_model.pth"
model = ConvNeXt(num_classes=5, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])

model_weight = torch.load(model_path)
model.load_state_dict(model_weight)

input_tensor = torch.ones((1, 3, 224, 224))
input_names = ['input_0']
output_names = ['output_0']

torch.onnx.export(
    model,
    input_tensor,
    'logs/best_model.onnx',
    opset_version=11,
    input_names=input_names,
    output_names=output_names
)
print('ONNX model saved successfully!!!')
