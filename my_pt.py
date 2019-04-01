import csv
import copy
from pathlib import Path
import shutil
import natsort
from glob import glob

def csv_out(id,dataset, model, epoch,rlt,val_scale):
    ori = copy.copy(rlt)
    rlt.sort(reverse=True)
    rlt_csv = []
    idx_csv = []
    for i in range(0, 2):
        rlt_csv.append("%.7f" % (rlt[i]))
        idx_csv.append((ori.index(rlt[i])+1)*val_scale)

    out_str = [id,dataset, model, epoch]
    out_str.extend(rlt_csv)
    out_str.extend(idx_csv)
    with open("out.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(out_str)

def leave_10(rlt):
    ori=copy.copy(rlt)
    rlt.sort(reverse=True)
    rlt_list = []
    idx_list = []
    for i in range(10):
        rlt_list.append("%.5f" % (rlt[i]))
        idx_list.append(ori.index(rlt[i]))

    IMG_File = natsort.natsorted(list(glob("checkpoints/*/")), alg=natsort.PATH)
    IMG_Str = []
    for i in IMG_File:
        IMG_Str.append(str(i))
    for k in range(len(IMG_Str)):
        c = int(IMG_Str[k].split("/")[1])
        if c not in idx_list:
            shutil.rmtree(IMG_Str[k])