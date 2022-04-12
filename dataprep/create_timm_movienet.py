import sys
import os
import numpy as np
import json


categories = ["ECS", "CS", "MS", "FS", "LS"]


def readAnnotations(filename):

    f = open(filename)
    srcdata = json.load(f)   

    return srcdata

# main

if len(sys.argv)<4:
    print("usage: create_timm_movienet.py annotations split outputdir sourcedir")
    exit()

annotationfilename = sys.argv[1]
split = sys.argv[2]
basedir = sys.argv[3]
imgsourcedir = sys.argv[4]

allannos = readAnnotations(annotationfilename)
keylist = list(allannos.keys())

if split=='train':
    keylist = keylist[:4843]
elif split == 'val':
    keylist = keylist[4843:5905]
else:
    keylist = keylist[5905:]

annos = {}
for k in keylist:
    annos[k] = allannos[k]


# create directories
for c in categories:
    os.makedirs(basedir + '/' + c,exist_ok = True)
    
for v in annos.keys():
    for s in annos[v].keys():
        mydir = annos[v][s]['scale']['label']
       
    srcname = v+'/shot_'+s+'.mp4.jpeg'
    trgname = v+'_shot_'+s+'.mp4.jpeg'

    if os.path.exists(imgsourcedir+'/'+srcname):
        os.symlink(imgsourcedir+'/'+srcname,basedir+'/'+mydir+'/'+trgname)
    else:
        print(imgsourcedir+'/'+srcname + ' does not exist')

