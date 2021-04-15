import onnx

test = onnx.load('alexnet.onnx')
onnx.checker.check_model(test)
print("==> Passed")
print(onnx.helper.printable_graph(test.graph))