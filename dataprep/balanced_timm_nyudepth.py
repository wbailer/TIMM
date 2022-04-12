import sys
import os
import numpy as np
import glob
import random


# main

if len(sys.argv)<3:
    print("usage: balanced_timm_nydepth.py srcdir trgdir [nItems]")
    print("       note: this script uses the directories containing JPEG files as input")
    print("       if optional parameter nItems is provided, this value will be used to limit classes with more samples, otherwise the minimum of all")
    exit()

srcdir = sys.argv[1]
trgdir = sys.argv[2]

subdirs = glob.glob(srcdir+'/*')

if len(sys.argv)==4:
    minNFiles = int(sys.argv[3])
else:
    #get counts
    minNFiles = 0
    for s in subdirs:
        jpegs = glob.glob(s+'/*.jpg')
        if len(jpegs)==0:
            print('warning: '+s+' contains no files')    
        if minNFiles==0 or len(jpegs)<minNFiles:
            minNFiles = len(jpegs)
        
if minNFiles==0:
    print('one directory is empty, min number of files is this 0')
    exit()
 
for s in subdirs:
    jpegs = glob.glob(s+'/*.jpg')
    
    random.shuffle(jpegs)
    jpegs = jpegs[:minNFiles]

    for img in jpegs:
       path = '/'.join(img.split('/')[:-1])
       path = trgdir+'/'+path[len(srcdir):]
       os.makedirs(path, exist_ok=True)
           
       os.symlink(img,trgdir+'/'+img[len(srcdir):])
