import cv2 as cv
from get_score import util
import numpy as np
import time
from pathlib import Path
import natsort

from get_score.metrics_my import runningScore
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

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

def load_image(path):
    image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
    return image

Tensor_Path = Path("/home/ali/cws/pytorch-semseg-dvs/test_out/deeplabv3/tensors")
Tensor_File = natsort.natsorted(list(Tensor_Path.glob("*.npy")), alg=natsort.PATH)
Tensor_Str = []
for j in Tensor_File:
    Tensor_Str.append(str(j))

GT_Path = Path("/home/ali/cws/pytorch-semseg-dvs/dataset/binary_resize_250")
GT_File = natsort.natsorted(list(GT_Path.glob("*.bmp")), alg=natsort.PATH)
GT_Str = []
for j in GT_File:
    GT_Str.append(str(j))

th=0.98
pre=[]
rec=[]
t = time.time()
running_metrics_val = runningScore(2)
label_values = [[0, 0, 0], [250, 250, 250]]
# label_values = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0]]

def compute_one(img_path,gt_path):
    out = load_image(img_path)
    gt = load_image(gt_path)
    # val_gt_erode paired with [0,0,0]label value
    # label order: R G B
    # num_classes = len(label_values)
    gt = util.reverse_one_hot(util.one_hot_it(gt, label_values))
    output_image = util.reverse_one_hot(util.one_hot_it(out, label_values))
    running_metrics_val.update(gt, output_image)


def full_one():
    running_metrics_val.reset()
    for k in range(len(Tensor_Str)):
        lanes_one_channel = np.load(Tensor_Str[k])
        pred = np.zeros((256, 512), dtype=np.uint)
        pred[lanes_one_channel > th] = 4
        decoded = decode_segmap(pred)
        out_path = "/home/ali/cws/pytorch-semseg-dvs/test_out/temp/" + Path(Tensor_Str[k]).stem + ".bmp"
        decoded_bgr = cv.cvtColor(decoded, cv.COLOR_RGB2BGR)
        # misc.imsave(out_path, decoded)
        cv.imwrite(out_path, decoded_bgr)

    IMG_Path = Path("/home/ali/cws/pytorch-semseg-dvs/test_out/temp")
    IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
    IMG_Str = []
    for i in IMG_File:
        IMG_Str.append(str(i))

    for k in range(len(IMG_Str)):
        compute_one(IMG_Str[k], GT_Str[k])

full_one()

# pool=ThreadPool(16)
# pool.starmap(compute_one,zip(IMG_Str,GT_Str))
# pool.close()
# pool.join()

acc, cls_pre, cls_rec, cls_f1, cls_iu, hist = running_metrics_val.get_scores()
tt = time.time() - t
pre.append(cls_pre)
rec.append(cls_rec)

print("cls pre")
print(cls_pre)
print("cls rec")
print(cls_rec)
print("time: %f" %tt)
