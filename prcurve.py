import numpy as np
from tqdm import tqdm
from pathlib import Path
import natsort
import cv2 as cv


def decode_segmap(temp):
    Imps = [0, 0, 0]
    Building = [100, 100, 100]
    Lowvg = [150, 150, 150]
    Tree = [200, 200, 200]
    Car = [250, 250, 250]
    # bg = [255,0,0]

    label_colours = np.array(
        [
            Imps,
            Building,
            Lowvg,
            Tree,
            Car,
            # bg,
        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 5):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]
    # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

Tensor_Path = Path("test_out/deeplabv3/tensors")
Tensor_File = natsort.natsorted(list(Tensor_Path.glob("*.npy")), alg=natsort.PATH)
Tensor_Str = []
for j in Tensor_File:
    Tensor_Str.append(str(j))

th=0.98
for k in tqdm(range(len(Tensor_Str))):
    lanes_one_channel=np.load(Tensor_Str[k])
    pred=np.zeros((256,512),dtype=np.uint)
    pred[lanes_one_channel>th]=4
    decoded = decode_segmap(pred)
    out_path="test_out/deeplabv3/prcurve/098/"+Path(Tensor_Str[k]).stem+".bmp"
    decoded_bgr = cv.cvtColor(decoded, cv.COLOR_RGB2BGR)
    # misc.imsave(out_path, decoded)
    cv.imwrite(out_path, decoded_bgr)
    # print("kk")



