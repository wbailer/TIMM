import sys
import os
import numpy as np



def readCatFile(name):

    index = {}

    f=open(name,"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        tokens = x.split(' ')
        index[int(tokens[1])] = tokens[0]
    f.close()
    return index

def readFileList(name):

    index = {}

    f=open(name,"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        tokens = x.split(' ')
        index[tokens[0]] = int(tokens[1])
    f.close()
    return index
    
def readSupercatList(name):

    index = {}
    mapping = np.zeros((20,365))

    f=open(name,"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        tokens = x.split(';')
        index[int(tokens[1])] = tokens[0]
        catmap = tokens[2].split(',')
        catmap = np.array(catmap).astype(int)
        mapping[int(tokens[1]),:] = catmap
    f.close()
    return (index,mapping)

# main

if len(sys.argv)<5:
    print("usage: create_timm_places365.py task categories filelist outputdir sourcedir [supercats level]")
    print("       task: base or sc, if sc the supercategories file and level must be present")
    exit()

task = sys.argv[1]
categories = readCatFile(sys.argv[2])
imgfiles = readFileList(sys.argv[3])
basedir = sys.argv[4]
imgsourcedir = sys.argv[5]
if task=='sc':
    supercatfile = sys.argv[6]
    level = int(sys.argv[7])
    create_sc = True
else:
    create_sc = False

categories_same_level = {}
for c in categories.keys():
    tokens = categories[c].split('/',2)
    csl = tokens[0]+'/'+tokens[1]+'/'+tokens[2].replace('/','_')
    categories_same_level[c] = csl


if len(sys.argv)>6:
    create_sc = True
    (supercats,catmap) = readSupercatList(supercatfile)

if (task=='base') and not(create_sc):
    # create directories
    for c in categories.keys():
        os.makedirs(basedir + categories_same_level[c])
    
    for v in imgfiles.keys():
        mydir = categories_same_level[imgfiles[v]]
        tokens = v.split('/')
        trgname = tokens[-1]
        if os.path.exists(imgsourcedir+'/'+v):
            os.symlink(imgsourcedir+'/'+v,basedir+mydir+'/'+trgname)
        else:
            print(imgsourcedir+'/'+v + 'does not exist')
                
if create_sc:
    scrange = []
    if level==0:
        scrange = [0,1]
    elif level==1:
        scrange = [0,2,3]
    else:
        scrange = list(range(4,len(supercats)))
    for idx in list(supercats.keys()):
        if not(idx in scrange):
            continue
        scdirname=supercats[idx].replace('(','').replace(')','').replace(',','').replace('.','').replace(' ','_')
        os.makedirs(basedir+'/'+scdirname)
        for v in imgfiles.keys():
            mydir = categories_same_level[imgfiles[v]]
            if catmap[idx,imgfiles[v]] > 0:
               trgname = v.replace('/','_')
               if os.path.exists(imgsourcedir+'/'+v):
                   os.symlink(imgsourcedir+'/'+v,basedir+'/'+scdirname+'/'+trgname)

    

