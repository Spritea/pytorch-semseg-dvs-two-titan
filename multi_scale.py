import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

import yaml
from pathlib import Path
import natsort


def test(args,cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # IMG_Path=Path(args.img_path)
    # IMG_File=natsort.natsorted(list(IMG_Path.glob("*.png")),alg=natsort.PATH)
    # IMG_Str=[]
    # for i in IMG_File:
    #     IMG_Str.append(str(i))
    # # Setup image
    # print("Read Input Image from : {}".format(args.img_path))

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset,config_file=cfg)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    # Setup Model
    # model = get_model(cfg['model'], n_classes)
    # state = convert_state_dict(torch.load(args.model_path)["model_state"])
    # # state=torch.load(args.model_path)["model_state"]
    # model.load_state_dict(state)
    # model.eval()
    # model.to(device)

    S2_Path=Path(args.img_path)
    S2_File=natsort.natsorted(list(S2_Path.glob("*S2.pt")),alg=natsort.PATH)
    S2_Str=[]
    for i in S2_File:
        S2_Str.append(str(i))

    S175_Path = Path(args.img_path)
    S175_File = natsort.natsorted(list(S175_Path.glob("*S175.pt")), alg=natsort.PATH)
    S175_Str = []
    for i in S175_File:
        S175_Str.append(str(i))

    S5_Path = Path(args.img_path)
    S5_File = natsort.natsorted(list(S5_Path.glob("*S5_ave.pt")), alg=natsort.PATH)
    S5_Str = []
    for i in S5_File:
        S5_Str.append(str(i))

    S4_Path = Path(args.img_path)
    S4_File = natsort.natsorted(list(S4_Path.glob("*S4d.pt")), alg=natsort.PATH)
    S4_Str = []
    for i in S4_File:
        S4_Str.append(str(i))

    S4_not_1_Path = Path(args.img_path)
    S4_not_1_File = natsort.natsorted(list(S4_not_1_Path.glob("*S4_not_1.pt")), alg=natsort.PATH)
    S4_not_1_Str = []
    for i in S4_not_1_File:
        S4_not_1_Str.append(str(i))

    F_Path = Path(args.img_path)
    F_File = natsort.natsorted(list(F_Path.glob("*F.pt")), alg=natsort.PATH)
    F_Str = []
    for i in F_File:
        F_Str.append(str(i))

    F_sum_Path = Path(args.img_path)
    F_sum_File = natsort.natsorted(list(F_sum_Path.glob("*F_sum.pt")), alg=natsort.PATH)
    F_sum_Str = []
    for i in F_sum_File:
        F_sum_Str.append(str(i))

    R4_sum_Path = Path(args.img_path)
    R4_sum_File = natsort.natsorted(list(R4_sum_Path.glob("*R4.pt")), alg=natsort.PATH)
    R4_sum_Str = []
    for i in R4_sum_File:
        R4_sum_Str.append(str(i))

    for j in tqdm(range(len(R4_sum_Str))):
        parts=Path(R4_sum_Str[j]).stem.split('_')
        parts.pop(-1)
        icon='_'
        img_name=icon.join(parts)
        # multi_avg = torch.zeros((1, 6, 512, 512), dtype=torch.float32).to(device)
        # sum=torch.load(S2_Str[j]).to(device)+torch.load(S175_Str[j]).to(device)+torch.load(S5_Str[j]).to(device)
        # sum=torch.load(S4_not_1_Str[j]).to(device)+torch.load(F_sum_Str[j]).to(device)
        # sum=torch.load(F_sum_Str[j]).to(device)+torch.load(S5_Str[j]).to(device)
        sum=torch.load(R4_sum_Str[j]).to(device)
        outputs=sum/4
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

        decoded = loader.decode_segmap(pred)
        out_path="test_out/mv3_1_true_2_res50_data10_MS/R4_images/"+img_name+".png"
        misc.imsave(out_path, decoded)

    # print("Classes found: ", np.unique(pred))
    # print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="pretrain/train16/mv3_res50_my_best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="my",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--img_path", nargs="?", type=str,
        default="test_out/mv3_1_true_2_res50_data10_MS/rotate_tensors", help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default="tk.png",
        help="Path of the output segmap",
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/mv3_1_true_2_res50_data10.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    test(args,cfg)
