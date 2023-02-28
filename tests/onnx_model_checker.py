import onnx
#load onnx model
model=onnx.load('D:/Dev Projects/2023 Projects/proto.landmark/src/landmark.onnx')

#check that model is well formed
onnx.checker.check_model(model)
#display human representation of graph

print(onnx.helper.printable_graph(model.graph))