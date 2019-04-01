import numpy as np
import cv2 as cv
from get_score import util
import time
from pathlib import Path
import natsort
# pre=np.load("/home/ali/cws/pytorch-semseg-dvs/get_score/scnn/pre.npy")
rec=np.load("/home/ali/cws/pytorch-semseg-dvs/get_score/scnn/rec.npy")


Tensor_Path = Path("/home/ali/cws/pytorch-semseg-dvs/test_out/scnn/tensors")
Tensor_File = natsort.natsorted(list(Tensor_Path.glob("*.npy")), alg=natsort.PATH)
Tensor_Str = []
for j in Tensor_File:
    Tensor_Str.append(str(j))

for k in range(len(Tensor_Str)):
    tensor=np.load(Tensor_Str[k])
    np_sum=tensor[:,:,0]+tensor[:,:,1]+tensor[:,:,2]+tensor[:,:,3]
    out_path_pt = "/home/ali/cws/pytorch-semseg-dvs/test_out/scnn/tensors_sum/" + Path(Tensor_Str[k]).name
    np.save(out_path_pt,np_sum)

print("kk")
