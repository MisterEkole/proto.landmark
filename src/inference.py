''' Pytorch script for model inferecing'''
from __future__ import print_function, division
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from math import *
from imutils import face_utils
import torchvision.transforms.functional as TF
import time
plt.ion()
import dlib
from network import *
import numpy as np

def load_model(model_path):
  model=XceptionNet()
  model.load_state_dict(torch.load(model_path, map_location='cpu'))
  #model=torch.load(model_path, map_location='cpu')
  model.eval()
  return model

model=load_model('D:/Dev Projects/DeepStack/proto.facelandmarkdetector/model/model_best.pth')

def transform_img(image):
  image=TF.to_pil_image(image)
  image = TF.resize(image, (224, 224))
  image = TF.to_tensor(image)
  image = (image - image.min())/(image.max() - image.min())
  image = (2 * image) - 1
  return image.unsqueeze(0)


def landmarks_draw(image,img_landmarks):
  image=image.copy()
  for landmarks, (left, top,height,width) in img_landmarks:
    landmarks=landmarks.view(-1,2)
    landmarks=(landmarks+0.5)
    landmarks=landmarks.numpy()

    for i, (x,y) in enumerate(landmarks, 1):
      try:
        cv2.circle(image, (int((x * width) + left), int((y * height) + top)), 2, [40, 117, 255], -1)
      except:
        pass
  return image


detector=dlib.get_frontal_face_detector()

@torch.no_grad()


def inference(frame):
  gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  
  faces=detector(gray,1)

  outputs=[]

  
  for (i, face) in enumerate(faces):
    (x, y, w, h) = face_utils.rect_to_bb(face)
    crop_img = gray[y: y + h, x: x + w]
    
    transformed_img= transform_img(crop_img)

    landmarks_predictions = model(transformed_img.cpu())

    outputs.append((landmarks_predictions.cpu(), (x, y, h, w)))
  return landmarks_draw(frame, outputs)


''' inference on video'''

def output_video(video, name, seconds = None):
    start_time=time.time()
    total = int(video.fps * seconds) if seconds else int(video.fps * video.duration)
    print('Will read', total, 'images...')
    
    outputs = []

    writer = cv2.VideoWriter(name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.fps, tuple(video.size))

    for i, frame in enumerate(tqdm(video.iter_frames(), total = total), 1):    
        if seconds:
            if (i + 1) == total:
                break
                
        output = inference(frame)
        outputs.append(output)

        

        writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    end_time=time.time()
    print("Model inference Time taken: {:.2f} seconds".format(end_time-start_time))

    

    writer.release()

    return outputs


''' inference on image'''

def image_output(image, name):
  print("Analysing image")
 

  outputs=[]



  output=inference(image)
  outputs.append(output)

  vec=np.empty((68,2), dtype=int)
  for b in range(68):
    vec[b][0]=output[0][b][0]
    vec[b][1]=output[0][b][1]

  print(vec)
  
  disp_img=cv2.imwrite(name + '.jpg', output)


  #writer.release()

  
  return outputs, disp_img



  



img_path='D:/Dev Projects/DeepStack/proto.facelandmarkdetector/data/faces/144044282_87cf3ff76e.jpg'
img=cv2.imread(img_path)
outputs=image_output(img,'D:/Dev Projects/DeepStack/proto.facelandmarkdetector/demo/img_output')
