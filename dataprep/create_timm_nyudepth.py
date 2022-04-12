import sys
import os
import numpy as np
import glob


def readCatFile(name):

    index = {}
    catlist = []

    f=open(name,"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        x = x.replace('\n','')
        tokens = x.split('\t')
        index[tokens[1]] = tokens[0]
        catlist.append(tokens[0])
    f.close()
    catlist = list(set(catlist))
    return (index,catlist)


# main

if len(sys.argv)<5:
    print("usage: create_timm_nydepth.py categories outputdir sourcedir valsplit")
    print("       valsplit: fraction of images to be put to validation")
    exit()

cats = sys.argv[1]
basedir = sys.argv[2]
imgsourcedir = sys.argv[3]
valsplit = float(sys.argv[4])
catmapping,catlist = readCatFile(cats)


# create directories
for c in catlist:
    os.makedirs(basedir + '/train/' + c)
    os.makedirs(basedir + '/val/' + c)
 
# create links   
for sc in catmapping.keys():
    rgbimages = glob.glob(imgsourcedir+'/'+sc+'/r-*.ppm')
    
    for img in rgbimages:
        tokens = img.split('/')
        trgname = tokens[-1]
        
        split = np.random.choice([0,1],p=[1-valsplit,valsplit])
        splitname = 'train'
        if split==1:
            splitname = 'val'

        # check for duplicates (some archives contain overlapping images)
        if not(os.path.exists(basedir+'/'+splitname+'/'+catmapping[sc]+'/'+trgname)):
            os.symlink(img,basedir+'/'+splitname+'/'+catmapping[sc]+'/'+trgname)

    
