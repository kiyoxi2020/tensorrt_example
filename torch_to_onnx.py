import torch
import torchvision

def get_model():
    model = torchvision.models.alexnet(pretrained=True).cuda()
    resnet_model = model.eval()
    return model


def get_onnx(model, onnx_save_path, example_tensor):
    example_tensor = example_tensor.cuda()
    _ = torch.onnx.export(model, 
                        example_tensor, 
                        onnx_save_path,
                        verbose=True,
                        training=False,
                        do_constant_folding=False,
                        input_names=['input'],
                        output_names=['output'])

if __name__ == '__main__':
    model = get_model()
    onnx_save_path = "alexnet.onnx"
    example_tensor = torch.randn(1, 3, 224, 224, device='cuda')

    get_onnx(model, onnx_save_path, example_tensor)
