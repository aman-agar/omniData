from flask import Flask,request,send_file
import os
import torch
import torch.nn.functional as F
from torchvision import transforms

from io import BytesIO
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

app=Flask(__name__)

@app.route('/depth',methods=['GET','POST'])
def depth():
    file = request.files['image']
    img=Image.open(file.stream)
    print("Image read successfully")
    # Call functions for getting depth and normal of input image
    depthImg=get_depth(img)
    # normalImg=get_normal(img)
    
    # Return both depthImg and NormalImg as output from API
    print("Successfully sending the output")
    print(type(depthImg))
    
    return returnPIL(depthImg)

def returnPIL(img):
    
    # img=Image.fromarray(img)
    if img.mode !='RGB':
        img=img.convert('RGB')
    img_io = BytesIO()
    # img_io=Image.fromarray(img_io)
    # if img_io.mode !='RGB':
        # img_io=img_io.convert('RGB')
    img.save(img_io, 'JPEG', quality=70, cmap='viridis')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/')
def trial():
    return "Use /depth in url to use depth and normal image"

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    print(tensor)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def get_depth(img): # Get img to work upon and Returns the depth output

    root_dir=root_dir = 'pretrained_model/'
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  
    
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path,map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    # model.to(device)
    
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])
    trans_rgb = transforms.Compose([transforms.Resize(512, 
                                    interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(512)])
    
    # Convert the image to tensor
    img_tensor = trans_totensor(img)[:3].unsqueeze(0) 
    print("Converted to tensor")
    
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3,1)
    
    # Get output from model
    output = model(img_tensor).clamp(min=0, max=1)
    print("Output Calculated")

    output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
    output = output.clamp(0,1)
    output = 1 - output
    print(type(output))

    output=output.detach().cpu().squeeze()
    print(output.shape) # Got 512x512 tensor 
    return tensor_to_image(output)



def get_normal(img): # Gets an image and Returns normal output
    root_dir=root_dir = 'pretrained_model/'
    image_size = 384
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    
    model.load_state_dict(state_dict)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        get_transform('rgb', image_size=None)])
    trans_rgb = transforms.Compose([transforms.Resize(512, 
                                    interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(512)])
    # Convert img to tensor
    img_tensor = trans_totensor(img)[:3].unsqueeze(0) 
    
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3,1)

    # Get output from model
    output = model(img_tensor).clamp(min=0, max=1)
    
    trans_topil = transforms.ToPILImage()
    return trans_topil(output[0])



if __name__=='__main__':
    app.run(debug=True,port=7000)
