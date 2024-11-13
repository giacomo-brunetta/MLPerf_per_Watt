import torch
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to('cuda')
dummy_input = torch.randn(1, 3, 224, 224).to('cuda')
torch.onnx.export(model, dummy_input, "resnet50.onnx", opset_version=13)