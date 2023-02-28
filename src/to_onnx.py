import torch
from network import *



def load_model(model_path):
  model=XceptionNet()
  model.load_state_dict(torch.load(model_path, map_location='cpu'))
  #model=torch.load(model_path, map_location='cpu')
  model.eval()
  return model

model=load_model('D:/Dev Projects/2023 Projects/proto.landmark/model/model_best.pth')


dum_input= torch.rand(32,1,224,224, device='cpu')
input_names=["actual_input1"]+["learned_%d" % i for i in range(16)]
output_names=["output1"]

torch.onnx.export(model,dum_input, 'landmark.onnx', verbose=True,input_names=input_names,output_names=output_names)

