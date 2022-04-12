import sys
import os
import numpy as np
import argparse
import json
import cv2
import copy
import csv
import glob
import math
import random
import jsonlines
import shutil
import uuid

# main

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', choices=['process_detections','process_facepose','create_ecu','stats','sample','samplemin','cvatimport','cvat2csv','timm','eval'],
                        required=True,
                        help='Task to be performed. '
                             'process_detections = process object detection, write (un)populated annotations, generate input for face and pose detection'
                             'process_facepose = process face and pose results, generate shot type annotations '
							 'create_ecu = create extreme close-up samples'
							 'stats = determine class statistics for file'
							 'sample = sample set for training/validation'
							 'samplemin = samplemin, same as sample, but minimising the overall of images'
							 'cvatimport = prepare JSON for CVAT import'
							 'cvat2csv = export from CVAT task file to CSV'
							 'timm = prepare structure for classification using Python Image Models (TIMM) framework'
							 'eval = evaluate annotationfile2 against annotationfile')
	
    parser.add_argument('--objjsonfile', type=str, default='',
                        help='Path to the COCO format JSON file with object detections to process'
                             'For face/pose processing, this is the file with the largest person per image')

    parser.add_argument('--personjsonfilepattern', type=str, default='',
                        help='For face/pose processing, this is the pattern for the set of JSON files for the largest person per image')
		 
    parser.add_argument('--objjsondir', type=str, default='',
                        help='Path to a directory with COCO format JSON files with object detections to be processed')
						
    parser.add_argument('--facefile', type=str, default='',
                        help='Path to the CSV file with face detections to process. If personjsonfilepattern is used, the number will be inserted to the face file name')

    parser.add_argument('--annotationfile', type=str, default='',
                        help='Path to the CSV file with class annotation (for stats, sample, CVAT, TIMM, eval) or JSON CVAT export (for CVAT2CSV)')

    parser.add_argument('--annotationfilepattern', type=str, default='',
                        help='Pattern to the CSV file with class annotation (for sample)')

							 
    parser.add_argument('--annotationfile2', type=str, default='',
                        help='Path to the CSV file with class annotation (for eval)')
							 
    parser.add_argument('--posefile', type=str, default='',
                        help='Path to the CSV file with pose detections to process. If personjsonfilepattern is used, the number will be inserted to the pose file name')
	
    parser.add_argument('--pos-thresh',type=float,default='0.1',
                        help='Confidence threshold to treat annotation as positive sample')

    parser.add_argument('--img-prefix',type=str,default='',
                        help='Prefix to strip from image names')
						
    parser.add_argument('--showimg',action='store_true',default=False,
                        help='Show detection preview')
						
    parser.add_argument('--showclass',action='store_true',default=False,
                        help='Show classification preview')
						
    parser.add_argument('--imgbasepath',type=str,default='',
                        help='Base path of images (for visualisation), also used for TIMM data setup')

    parser.add_argument('--outdir',type=str,default='.',
                        help='Output directory. Note: face/pose processing will read popcls.csv from out create a file extended with shot types')
						
    parser.add_argument('--imgoutdir',type=str,default=None,
                        help='Image output directory, for creating extreme closeup shots only, also used for TIMM data setup')
						
    parser.add_argument('--anno-discarded',action='store_true',default=False,
                        help='Annotate also discarded files')

    parser.add_argument('--numperclass',type=int,default=100,
                        help='For sampling and TIMM: the number of target samples per class')

    parser.add_argument('--imagelist',type=str,
                        help='For CVAT: the image index from a task export')

    parser.add_argument('--classtype',type=str,
                        help='For TIMM: type of the classification task: pop, shot')

    parser.add_argument('--shotcls',type=str,default=None,
                        help='For sample: only comma separated subset of shot classes')

    parser.add_argument('--popcls',type=str,default=None,
                        help='For sample: only comma separated subset of populated classes')

    parser.add_argument('--exclude',type=str,default=None,
                        help='For sample: exclude images listed in the specified CSV file')

    parser.add_argument('--binary-pop',action='store_true',default=False,
                        help='For TIMM: binary populated class')

    parser.add_argument('--acc-pm1',action='store_true',default=False,
                        help='For eval: count classification to class +/1 as correct')
						
    args = parser.parse_args()
    return args

def loadFilterFile(filename,pos_thresh,prefix):

    f = open(filename)
    srcdata = json.load(f)   
	
    imgdict = {}
	
    if len(prefix)>0:
        plen = len(prefix)
        for i in srcdata['images']:
            prefidx	= i['file_name'].find(prefix)
            if prefidx>-1:
                i['file_name'] = i['file_name'][plen:]
            imgdict[i['file_name']] = i
	
    keptannos = []
    posimages = {}
    discardimages = {}
	
	# person and vehicles to be considered
    # 1, person
    # 2, bicycle
    # 3, car
    # 4, motorcycle
    # 5, airplane
    # 6, bus
    # 7, train
    # 8, truck
    # 9, boat
    classes_of_interest = [1,2,3,4,5,6,7,8,9]
	
    for a in srcdata['annotations']:
	    # FIX ID
        a['category_id'] = a['category_id'] + 1
        if a['category_id'] in classes_of_interest:
            if a['score']>pos_thresh:			
                keptannos.append(a)
                img = srcdata['images'][a['image_id']]
                posimages[img['file_name']] = img
            else:
                img = srcdata['images'][a['image_id']]
                discardimages[img['file_name']] = img
			
	
    negimages = {}
    for i in imgdict.keys():
        found = False
        if not(i in posimages.keys()) and not(i in discardimages.keys()):
            negimages[i] = imgdict[i]
		
    data = {}
    data['info'] = srcdata['info']
    data['licenses'] = srcdata['licenses']
    data['images'] = srcdata['images']
    data['categories'] = srcdata['categories']
    data['images_pos'] = posimages
    data['images_neg'] = negimages
    data['images_discard'] = discardimages
    data['annotations'] = keptannos

    return data    
	
def displayImages(imgdict,annos,basepath):	

    for imgname in imgdict.keys():

        img = cv2.imread(basepath + '/' + imgname)
	
        mask = np.zeros(img.shape,dtype=np.uint8)

        for item in annos:
            if item['image_id'] == imgdict[imgname]['id']:
                points = [ (item['bbox'][0], item['bbox'][1]) ]
                points.append( (item['bbox'][0], item['bbox'][1] + item['bbox'][3] ) )
                points.append( (item['bbox'][0] + item['bbox'][2], item['bbox'][1] + item['bbox'][3] ) )
                points.append( (item['bbox'][0] + item['bbox'][2], item['bbox'][1] ) )
                points = np.array(points).astype(np.int32)

                color = (255,0,0)
                if item['category_id'] == 1:
                    color = (0,0,255)
                cv2.fillConvexPoly(mask, points, color)

        jointimg = np.zeros((img.shape[0],img.shape[1]*2,img.shape[2]),dtype=np.uint8)
        jointimg[:,:img.shape[1],:] = img
        jointimg[:,img.shape[1]:,:] = mask

        cv2.imshow("imagedet", jointimg)
        cv2.waitKey()
		
def displayImageClass(imgdict,popcls,basepath):	

    for imgname in imgdict.keys():

        print(basepath + '/' + imgname)
	
        img = cv2.imread(basepath + '/' + imgname)
		
        position = (10,40)
        popstr = "unpopulated"
        if popcls[imgname]==1:
            popstr = "fewpeople"
        elif popcls[imgname]==2:
            popstr = "fewvehicles"
        elif popcls[imgname]==3:
            popstr = "fewlarge"
        elif popcls[imgname]==4:
            popstr = "medium"
        elif popcls[imgname]==5:
            popstr = "populated"
		
        cv2.putText(img, popstr, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 3)
		
        cv2.imshow("imageclass", img)
        cv2.waitKey()
		
def determinePopulatedClass(anno,images_pos,images_neg):

    # classes: 
	# 0: unpopulated
	# 1: few people: less than 3, area less than 10%
	# 2: few vehicles: less than 3, area less than 20%
	# 3: few large people/vehicles: less than 3, any area
	# 4: medium: less than 11 people/vehicles, area less than 30%
	# 5: populated

    classification = {}
	
    k = 0
	
    for imgname in images_pos.keys():
        numberP = 0
        numberV = 0
        areaP = 0
        areaV = 0
		
        imgarea = images_pos[imgname]['width']*images_pos[imgname]['height']
        for a in anno:
            if a['image_id'] == images_pos[imgname]['id']:
                area = a['bbox'][2]*a['bbox'][3]
                if a['category_id']==1:
                    numberP = numberP +1
                    areaP = areaP + area
                else:
                    numberV = numberV +1
                    areaV = areaV + area
        area = (areaP+areaV)/imgarea
        cls = 5
        if numberP<3 and numberV==0 and area<0.1:
            cls = 1
        elif numberV<3 and numberP==0 and area<0.2:
            cls = 2
        elif (numberP+numberV<3):
            cls = 3
        elif numberP+numberV<11 and area<0.3:
            cls = 4
        classification[imgname] = cls
		
        if k % 10000==0:
            print('   done '+str(k))
        k = k+1
	
    for imgname in images_neg.keys():
        classification[imgname] = 0
	
    return classification
	
	
def filterPersonAnnotations(anno,images_pos):

    minBBTh = 0.015 # BB is assumed to be at least 1.5% of image width

    personAnnos = {}
	
    k = 0
	
    for imgname in images_pos.keys():
        imw = images_pos[imgname]['width']
        largestPerson = -1
        largestPersonAnno = None
		
        for a in anno:
            if a['image_id'] == images_pos[imgname]['id']:
                if a['category_id']==1:
                    #area = a['bbox'][2]*a['bbox'][3]
                    height = a['bbox'][3]
                    if min(a['bbox'][2],a['bbox'][3])>minBBTh:
                        if height>largestPerson:
                            largestPerson = height
                            largestPersonAnno = a
						
							
        if largestPersonAnno is not None:
            personAnnos[imgname] = largestPersonAnno
		
        if k % 10000==0:
            print('   done '+str(k))
        k = k+1
		
    return personAnnos

	
def readCSV(filename):
    imgdict = {}

    rownr = 0
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # skip header
            rownr = rownr + 1
            if rownr == 1:
                continue
            imgdict[row[0]] = list(np.array(row[1:]).astype(np.int32))
            if len(imgdict[row[0]])==1:
                imgdict[row[0]] = imgdict[row[0]][0]
			
    return imgdict

	
def writeCSV(filename,imgdict,popcls=None,shotcls=None):

    header=['image']
    if popcls is not None:
        header.append('populated')
    if shotcls is not None:
        header.append('shottype')

    data = []
	
    for img in imgdict.keys():
        row = [img]
        if popcls is not None:
           row.append(popcls[img])
        if shotcls is not None:
           if img in shotcls.keys():
               row.append(shotcls[img])
           else:
               row.append(0)
        if len(data)==0:
            data = [row]
        else:
            data.append(row)

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
	
def writePersonJSON(filename,prefix,srcannos,personannos):

    annofile = {}
    annofile['info'] = srcannos['info']
    annofile['licenses'] = srcannos['licenses']
    annofile['categories'] = srcannos['categories']
	
    annofile['images'] = []
    annofile['annotations'] = []
	
    imgids = {}
	
    for a in personannos.items():
        a = a[1]
        imgids[a['image_id']]=''
        annofile['annotations'].append(a)	
		
		
    plen = len(prefix)
    for iid in imgids.keys():
        i = 0
        img = srcannos['images'][0]
        while not(int(iid)==int(img['id'])): 
            i = i + 1
            img = srcannos['images'][i]
        prefidx = img['file_name'].find(prefix)
        if prefidx>-1:
            img['file_name'] = img['file_name'][plen:]
        annofile['images'].append(img)
				
    with open(filename, 'w') as outfile:
        json.dump(annofile, outfile)
	
def processDetections(annos,args,i):

    discpopcls = None
    discpersonAnnos = None	

    if args.showimg:
        displayImages(annos['images_pos'],annos['annotations'],args.imgbasepath)
       	   
    print('creating populated annoations ...')
    popcls = determinePopulatedClass(annos['annotations'],annos['images_pos'],annos['images_neg'])
	 
    print('filtering person annotations ...')
    personAnnos = filterPersonAnnotations(annos['annotations'],annos['images_pos'])
	
    if args.showclass:
        allannos = copy.copy(annos['images_pos'])
        allannos.update(annos['images_neg'])
        displayImageClass(allannos,popcls,args.imgbasepath)
	
    suffix = str(i)
    if i < 0:
        suffix = ''
		
    if args.outdir is not None:
        print('writing populated annotations ... ')
        writeCSV(args.outdir+'/popcls'+suffix+'.csv',popcls,popcls)
        print('writing person annotations ... ')
        writePersonJSON(args.outdir+'/largestpersons'+suffix+'.json',args.img_prefix,annos,personAnnos)
			
    if (args.outdir is not None) and args.anno_discarded:
        discpopcls = determinePopulatedClass(annos['annotations'],annos['images_discard'],{})
        writeCSV(args.outdir+'/discarded_popcls'+suffix+'.csv',discpopcls,discpopcls)
						
        discpersonAnnos = filterPersonAnnotations(annos['annotations'],annos['images_discard'])
        writePersonJSON(args.outdir+'/discarded_largestpersons'+suffix+'.json',args.img_prefix,annos,discpersonAnnos)
	

    return popcls, personAnnos, discpopcls, discpersonAnnos
	
def preparePoseData(posedata, pdtype="smpl54"):

    if pdtype=="evo17":
        posedata = np.reshape(posedata,(17,2))
        jointIndex = {}
        jointIndex['pelvis'] = 0
        jointIndex['lfoot'] = 6
        jointIndex['rfoot'] = 3
        jointIndex['head'] = 10
        jointIndex['lhip'] = 4
        jointIndex['rhip'] = 1
        jointIndex['thorax'] = 8
        jointIndex['lknee'] = 5
        jointIndex['rknee'] = 2
        jointIndex['spine'] = 7
        return posedata,jointIndex
    elif pdtype=="smpl54":
        posedata = np.reshape(posedata,(54,2))
        jointIndex = {}
        jointIndex['pelvis'] = 0
        jointIndex['lfoot'] = 10
        jointIndex['rfoot'] = 11
        jointIndex['head'] = 53
        jointIndex['lhip'] = 1
        jointIndex['rhip'] = 2
        jointIndex['thorax'] = 6
        jointIndex['lknee'] = 4
        jointIndex['rknee'] = 5
        jointIndex['spine'] = 3
        return posedata,jointIndex	
    else:
        print("ERROR: unknown pose data type")
        return ([],{})


	
def processFacePoses(annos,faces,poses,args):

    shottypes = {}
	
    # 0 undefined
    # 1 extreme close-up
    # 2 close-up
    # 3 medium close-up
    # 4 tight medium shot
    # 5 medium shot
    # 6 medium full shot
    # 7 full shot
    # 8 long shot
    # 9 extreme long shot
    # 10 aerial view [NOTE: not yet supported]

    for a in annos['annotations']:
        
        imgid = a['image_id']
        imgname = ''
        imgheight = -1

        for img in annos['images']:
            if int(img['id']) == int(imgid):
                imgname = img['file_name']
                imgheight = img['height']
                break
				
        bbox = a['bbox']
		
        hasPose = False
        if imgname in poses.keys():
            hasPose = True
			 
            posedata,ji = preparePoseData(poses[imgname],"smpl54")
            
			# filter poses that fill only a small part of the longer side of the person box
            minFact = 0.6
			
            poseW = posedata[:,0].max()-posedata[:,0].min()
            poseH = posedata[:,1].max()-posedata[:,1].min()
            			
            if ((bbox[2]>bbox[3]) and (poseW < bbox[2]*minFact)) or ((bbox[3]>=bbox[2]) and (poseH < bbox[3]*minFact)):
                hasPose = False
            else:
                ignoreLegs = False
                # check if head and at least one foot are on different sides of pelvis
                pelvisHead = posedata[ji['pelvis'],1] - posedata[ji['head'],1]
                feetPelvis = max(posedata[ji['lfoot'],1],posedata[ji['rfoot'],1]) - posedata[ji['pelvis'],1]
                if math.copysign(1,pelvisHead)!=math.copysign(1,feetPelvis):
                    ignoreLegs = True
                # check if legs appear to be stretched (hip to feet > pelvis to thorax)
                if abs(max(posedata[ji['lfoot'],1]-posedata[ji['lhip'],1],posedata[ji['rfoot'],1]-posedata[ji['rhip'],1]))>abs(posedata[ji['pelvis'],1]-posedata[ji['thorax'],1]):
                    ignoreLegs = True
					
                if not(ignoreLegs): 
                    ignoreFeet = False
                    # check if feet and hip on different side of knee (at least for one foot)
                    hipKneeL = posedata[ji['lknee'],1] - posedata[ji['lhip'],1]
                    kneeFootL = posedata[ji['lfoot'],1] - posedata[ji['lknee'],1]
                    hipKneeR = posedata[ji['rknee'],1] - posedata[ji['rhip'],1]
                    kneeFootR = posedata[ji['rfoot'],1] - posedata[ji['rknee'],1]
                    if math.copysign(1,hipKneeL)!=math.copysign(1,kneeFootL) or math.copysign(1,hipKneeR)==math.copysign(1,kneeFootR):
                        ignoreFeet = True
	                    # assume 3/4 height to knee
                        assumedPoseBB = [bbox[0],bbox[1],bbox[2],int(4/3*abs(max(posedata[ji['lknee'],1],posedata[ji['rknee'],1])-posedata[ji['head'],1]))]
                        # scale to correct
                        scaledW = int(assumedPoseBB[2]*1.0)
                        scaledH = int(assumedPoseBB[3]*1.0)
                        assumedPoseBB[2] = scaledW
                        assumedPoseBB[3] = scaledH
                        assumedPoseBB[0] = int(assumedPoseBB[0]-(scaledW-assumedPoseBB[2])/2.0)
                        assumedPoseBB[1] = int(assumedPoseBB[1]-(scaledH-assumedPoseBB[3])/2.0)
                    else:
                        assumedPoseBB = bbox
                else:
                    # assume height as 2 pelvis - head
                    assumedPoseBB = [posedata[:,0].min(),posedata[:,1].min(),poseW,2*abs(pelvisHead)]
                    # scale to correct
                    scaledW = int(assumedPoseBB[2]*1.0)
                    scaledH = int(assumedPoseBB[3]*1.3)
                    assumedPoseBB[2] = scaledW
                    assumedPoseBB[3] = scaledH
                    assumedPoseBB[0] = int(assumedPoseBB[0]-(scaledW-assumedPoseBB[2])/2.0)
                    assumedPoseBB[1] = int(assumedPoseBB[1]-(scaledH-assumedPoseBB[3])/2.0)
			
        hasFace = False
        if imgname in faces.keys():
            hasFace = True		
        	# filter faces that fill only a small part of the person box
            minFact = 0.1
            facedata = np.array(faces[imgname])
            
            if (facedata[2] < bbox[2]*minFact) and (facedata[3] < bbox[3]*minFact):
                hasFace = False
            else:
                # assumption: head height ~ 1/8 of body height => face height ~ 1/10 of body height + add this offet on top
                # check if box should extend up or down
                height = 10*facedata[3]
                if facedata[1] < bbox[1]+bbox[3]*0.5:
                    top = int(facedata[1]-facedata[3]*0.5)
                else:
                    top = int(facedata[1]+facedata[3]-height)
                assumedFaceBB = [bbox[0],top,bbox[2],height]
	  
        print(imgname)
	
        if hasPose:
            # factor for pose adjustment due to face
            integratedBB = assumedPoseBB
            hipH = min(posedata[ji['lhip'],1],posedata[ji['rhip'],1])
            kneeH = min(posedata[ji['lknee'],1],posedata[ji['rknee'],1])
            if hasFace:
                integratedFaceBB = facedata
                # integrate BBs
                if ignoreLegs:
                    integratedBB[3] = assumedFaceBB[3]
                    kneeH = hipH + int(((integratedBB[1] + integratedBB[3])-hipH)/2)
                elif ignoreFeet:
                    integratedBB[3] = int((assumedPoseBB[3]+assumedFaceBB[3])/2)
            else:
                # predict face BB
                integratedFaceBB = [ integratedBB[0], integratedBB[1], integratedBB[2], int(integratedBB[3]/8) ]  
            shottypes[imgname] = classifyPersonHeight(imgheight,integratedBB[1],integratedBB[1]+integratedBB[3]-1,integratedFaceBB[1],integratedFaceBB[1]+integratedFaceBB[3]-1,
			                                              posedata[ji['thorax'],1],posedata[ji['spine'],1],posedata[ji['pelvis'],1],hipH,kneeH)
        elif hasFace:
            # estimate proportions
            integratedBB = assumedFaceBB
            integratedFaceBB = facedata
            thoraxH = integratedBB[1] + int(0.25*integratedBB[3])
            spineH = integratedBB[1] + int((3.0/8.0)*integratedBB[3])
            pelvisH = integratedBB[1] + int((3.5/8.0)*integratedBB[3])
            hipH = integratedBB[1] + int(0.5*integratedBB[3])
            kneeH = integratedBB[1] + int(0.75*integratedBB[3])
            shottypes[imgname] = classifyPersonHeight(imgheight,integratedBB[1],integratedBB[1]+integratedBB[3]-1,integratedFaceBB[1],integratedFaceBB[1]+integratedFaceBB[3]-1,thoraxH,spineH,pelvisH,hipH,kneeH)
        else:
            # check if person bounding box extends to bottom, otherwise estimate long/extreme long shot
            if bbox[1]+bbox[3] < 0.98*imgheight:
                if bbox[3] <= 0.25*imgheight:
                    shottypes[imgname] = 9
                elif bbox[3] > 0.6*imgheight:
                    shottypes[imgname] = 8
                else:
                    shottypes[imgname] = 0
            else:
                shottypes[imgname] = 0
	
        if args.showimg:
            img = cv2.imread(args.imgbasepath + '/' + imgname)


            cv2.rectangle(img,[bbox[0],bbox[1]],[bbox[0]+bbox[2],bbox[1]+bbox[3]],(255,255,255),5)
            if hasFace:
               cv2.rectangle(img,[assumedFaceBB[0],assumedFaceBB[1]],[assumedFaceBB[0]+assumedFaceBB[2],assumedFaceBB[1]+assumedFaceBB[3]],(0,0,255),3)
            if hasPose:
               cv2.rectangle(img,[assumedPoseBB[0],assumedPoseBB[1]],[assumedPoseBB[0]+assumedPoseBB[2],assumedPoseBB[1]+assumedPoseBB[3]],(255,0,0),1)
		
            cv2.imshow("rects", img)
            cv2.waitKey()
			   
            #print('decision '+str(shottypes[imgname]))
			   
            #cv2.imwrite('./estimsz/'+'.'.join((imgname.split('/')[-1]).split('.')[:-1])+'_shottype_'+str(shottypes[imgname])+'.jpg',img)
			
    return shottypes
        		
    
def classifyPersonHeight(imgH,personTop,personBottom,faceTop,faceBottom,thoraxH,spineH,pelvisH,hipH,kneeH):

    # print('imgH: '+str(imgH))
    # print('personTop: '+str(personTop))
    # print('personBottom: '+str(personBottom))
    # print('faceTop: '+str(faceTop))
    # print('faceBottom: '+str(faceBottom))
    # print('thoraxH: '+str(thoraxH))
    # print('spineH: '+str(spineH))
    # print('pelvisH: '+str(pelvisH))
    # print('hipH: '+str(hipH))
    # print('kneeH: '+str(kneeH))
    
    # 1 extreme close-up
    if (faceTop <= 0.05 * imgH) and (faceBottom >= 0.95 * imgH):
        return 1
    # 2 close-up		
    elif faceBottom <= imgH and spineH > imgH:
        return 2
    # 3 medium close-up
    elif spineH <= imgH and pelvisH > imgH:
        return 3
    # 4 tight medium shot
    elif pelvisH <= imgH and hipH > imgH:
        return 4
    # 5 medium shot
    elif hipH <= imgH and kneeH > imgH:
        return 5
    # 6 medium full shot
    elif kneeH <= imgH and personBottom > imgH:
        return 6
    # 7 full shot
    elif personBottom <= imgH and (personBottom-personTop)>0.6*imgH:
        return 7
    # 8 long shot
    elif (personBottom-personTop)>0.25*imgH:
        return 8
    # 9 extreme long shot
    else:
        return 9
	
def create_ecu(faces,popshotcls,args):

    popcls_ecu = {}
    shotcls_ecu = {} 

    for imgname in popshotcls:
	
        popcls_ecu[imgname] = popshotcls[imgname][0]
        shotcls_ecu[imgname] = popshotcls[imgname][1]
	
        # we are only interested in closeups
        if popshotcls[imgname][1] != 2:
            continue
			
        # make sure we have a face
        if imgname not in faces.keys():
            continue

        inputImgOrig = cv2.imread(args.imgbasepath+'/'+imgname)	        
		
        bbox = faces[imgname]
		
        minFaceSz = 170
		
        # check if face is large enough
        if bbox[2]<=minFaceSz or bbox[3]<=minFaceSz:
            continue
		
        # choose arbitrary box larger min minFaceSz, max 0.75 of face
        minSz = minFaceSz
        maxSz = max(minFaceSz,int(min(bbox[2],bbox[3])*.75))
		
        w = random.randint(minSz,maxSz)
        h = random.randint(minSz,maxSz)
		
        x = random.randint(bbox[0],bbox[0]+bbox[2]-w)
        y = random.randint(bbox[1],bbox[1]+bbox[2]-w) 
		
        subImg = np.copy(inputImgOrig[y:y+h-1,x:x+w-1, : ])
		
        #modimgname = '.'.join((imgname.split('/')[-1]).split('.')[:-1])
        modimgname = (imgname.split('.')[0]).replace('/','_') 
        if modimgname[0] == '_':
            modimgname = modimgname[1:]
		
        extImgName = '/'+modimgname+'_ecu.jpg'
		
        cv2.imwrite(args.imgoutdir+extImgName,subImg)
		
        popcls_ecu[extImgName] = 999 #invalid
        shotcls_ecu[extImgName] = 1 # extreme closeup
       
    return popcls_ecu, shotcls_ecu
	
def checktargetnum(popann,shotann,targetnum,popcls,shotcls):

    targetReached = 0

    for v in popcls:
        targetReached += targetnum - len(popann[v])
    
    for v in shotcls:
        targetReached += targetnum - len(shotann[v])
		
    return targetReached
		
def readJSONLines(filename,prefix=''):

    fileidx = {}

    idx = 0	
    with jsonlines.open(filename) as reader:
        for obj in reader:
            if 'name' not in obj.keys():
                continue
            fileidx[prefix+obj['name']+obj['extension']] = idx
            idx = idx +1		
		   
    return fileidx
	
def getLabelName(labelId,type):

    if type=='pop':
        if labelId==0:
            return 'unpopulated'		
        if labelId==1:
            return 'few people'		
        if labelId==2:
            return 'few vehicles'		
        if labelId==3:
            return 'few large'		
        if labelId==4:
            return 'medium'		
        if labelId==5:
            return 'populated'
        if labelId==999:
            return 'undefined'
    if type=='shot':
        if labelId==0:
            return 'undefined'		
        if labelId==1:
            return 'extreme close-up'		
        if labelId==2:
            return 'close-up'		
        if labelId==3:
            return 'medium close-up'		
        if labelId==4:
            return 'tight medium shot'		
        if labelId==5:
            return 'medium shot'
        if labelId==6:
            return 'medium full shot'		
        if labelId==7:
            return 'full shot'		
        if labelId==8:
            return 'long shot'		
        if labelId==9:
            return 'extreme long shot'		
        if labelId==10:
            return 'aerial view'    
			
def getLabelId(labelName,type):

    if type=='pop':
        if labelName=='unpopulated':
            return 0		
        if labelName=='few people':
            return 1		
        if labelName=='few vehicles':
            return 2		
        if labelName=='few large':
            return 3		
        if labelName=='medium':
            return 4		
        if labelName=='populated':
            return 5
        if labelName=='undefined':
            return 999
    if type=='shot':
        if labelName=='undefined':
            return 999		
        if labelName=='extreme close-up':
            return 1		
        if labelName=='close-up':
            return 2		
        if labelName=='medium close-up':
            return 3		
        if labelName=='tight medium shot':
            return 4		
        if labelName=='medium shot':
            return 5
        if labelName=='medium full shot':
            return 6		
        if labelName=='full shot':
            return 7		
        if labelName=='long shot':
            return 8		
        if labelName=='extreme long shot':
            return 9		
        if labelName=='aerial view':
            return 10   
    return 999			
	
def main(arglist):

    sys.argv = arglist
    sys.argc = len(arglist)

    args = parse_args()
	
	# single file for processing detection or pose/face
    if len(args.objjsonfile)>0:
        if args.task == 'process_detections':
            annos = loadFilterFile(args.objjsonfile,args.pos_thresh,args.img_prefix)
            popcls, personAnnos, discpopcls, discpersonAnnos = processDetections(annos,args,-1)
			
        elif args.task == 'process_facepose':
            f = open(args.objjsonfile)
            annos = json.load(f)  
		
            facedict = readCSV(args.facefile)
            posedict = readCSV(args.posefile)
			
            # if file was numbered, add number again
            number = ''
            filenameparts = args.facefile.split('/')
            filenamebase = filenameparts[-1].split('.') 
            k = -1
            while filenamebase[0][k:].isdigit():
                number = filenamebase[0][k:]
                k = k - 1
			
            popcls = readCSV(args.outdir + '/popcls'+number+'.csv')
			
            shottypes = processFacePoses(annos,facedict,posedict,args)
		
            writeCSV(args.outdir + '/popshotcls'+number+'.csv',popcls,popcls,shottypes)

	# batch processing for detections
    elif len(args.objjsondir)>0:
        i = 0
        popcls = {}
        discpopcls = {}
        personAnnos = {}
        discpersonAnnos = {}
        allannos = {}
        for fn in glob.glob(args.objjsondir+'/*.json'):
            annos = loadFilterFile(fn,args.pos_thresh,args.img_prefix)
            allannos['categories'] = annos['categories']
            allannos['info'] = annos['info']
            allannos['licenses'] = annos['licenses']
            if 'images' not in allannos['keys']:
                allannos['images'] = annos['images']
            else:
                allannos['images'].update(annos['images'])
            if args.task == 'process_detections':
        		
                popclsI, personAnnosI, discpopclsI, discpersonAnnosI = processDetections(annos,args,i)
                i = i + 1

                popcls.update(popclsI)
                if discpopclsI is not None:
                    discpopcls.update(discpopclsI)
					
                personAnnos.update(personAnnosI)
                if discpersonAnnosI is not None:
                    discpersonAnnos.update(discpersonAnnosI)
		
		
        if args.outdir is not None:
            print('writing populated annotations ... ')
            writeCSV(args.outdir+'/popcls.csv',popcls,popcls)
            print('writing person annotations ... ')
            writePersonJSON(args.outdir+'/largestpersons.json',args.img_prefix,annos,personAnnos)
			
        if (args.outdir is not None) and args.anno_discarded:
            discpopcls = determinePopulatedClass(annos['annotations'],annos['images_discard'],{})
            writeCSV(args.outdir+'/discarded_popcls.csv',discpopcls,discpopcls)
						
            discpersonAnnos = filterPersonAnnotations(annos['annotations'],annos['images_discard'])
            writePersonJSON(args.outdir+'/discarded_largestpersons.json',args.img_prefix,annos,discpersonAnnos)
        
    # batch processing for pose/face
    elif len(args.personjsonfilepattern)>0:
        files = glob.glob(args.personjsonfilepattern)
		
        overallpopcls = {}
        overallshotcls = {}
		
        for filename in files:
            f = open(filename)
            anno = json.load(f)   
			
            namepart = filename.split('/')[-1]
            namepart = namepart.split('\\')[-1]
            namepart = '.'.join(namepart.split('.')[:-1])
				
            nrsplit = len(namepart)
            while namepart[nrsplit-1:].isdigit():
                nrsplit = nrsplit -1
            nrpart = namepart[nrsplit:]
            namepart = namepart[:nrsplit]
	
            facefilename = '.'.join(args.facefile.split('.')[-1])			
            facedict = readCSV(facefilename+nrpart+'.csv')
			
            posefilename = '.'.join(args.facefile.split('.')[-1])			
            posedict = readCSV(posefilename+nrpart+'.csv')
			
            popcls = readCSV(args.outdir + '/popcls'+nrpart+'.csv')
			
            shottypes = processFacePoses(annos,facedict,posedict,args)
		
            writeCSV(args.outdir + '/popshotcls'+nrpart+'.csv',popcls,popcls,shottypes)
			
            overallpopcls.update(popcls)
            overallshotcls.update(shotcls)			
         
            writeCSV(args.outdir + '/popshotcls.csv',overallpopcls,overallpopcls,overallshotcls)
		 
    elif args.task == 'create_ecu':
        facedict = readCSV(args.facefile)
      
        popshotcls = readCSV(args.annotationfile)
			
        popcls_ecu, shotcls_ecu = create_ecu(facedict,popshotcls,args)
						
        # if file was numbered, add number again
        number = ''
        filenameparts = args.annotationfile.split('/')
        filenamebase = filenameparts[-1].split('.') 
        k = -1
        while filenamebase[0][k:].isdigit():
            number = filenamebase[0][k:]
            k = k - 1
		
        writeCSV(args.outdir + '/popshotcls_ecu'+number+'.csv',popcls_ecu,popcls_ecu,shotcls_ecu)

    elif args.task == 'stats':
	
        annotations = readCSV(args.annotationfile)
		
        popcls = [0,1,2,3,4,5,999]
        shotcls = [0,1,2,3,4,5,6,7,8,9,10]
	
        stats = {}
	
        # init keys
        for v in popcls:
            stats['p'+str(v)] = 0
        for v in shotcls:
            stats['s'+str(v)] = 0     
        for v1 in popcls:
            for v2 in shotcls:			
                stats[str(v1)+'_'+str(v2)] = 0 
			
        for a in annotations.keys():
            values = annotations[a]
            stats['p'+str(values[0])] += 1
            stats['s'+str(values[1])] += 1
            stats[str(values[0])+'_'+str(values[1])] += 1
			
        for s in stats.keys():
            print(s+':\t\t'+str(stats[s]))

			
    elif args.task == 'sample':

        if len(args.annotationfile)>0:
            annotations = readCSV(args.annotationfile)
			
        elif len(args.annotationfilepattern)>0:
            files = glob.glob(args.annotationfilepattern)
		
            annotations = {}
		
            for filename in files:
                f = open(filename)
                annotationsf = readCSV(filename)
                annotations.update(annotationsf)
        

        excludelist = []
		
        if args.exclude is not None:
            excludedannotations = readCSV(args.exclude)
            excludelist = excludedannotations.keys()
		
        # classes to sample (excluding undefined, aerial)
        popcls = [0,1,2,3,4,5]
		
        if args.popcls is not None:
            popcls = np.array(args.popcls.split(',')).astype(np.int32)
		
        shotcls = [1,2,3,4,5,6,7,8,9]

        if args.shotcls is not None:
            shotcls = np.array(args.shotcls.split(',')).astype(np.int32)

		
        targetnum = args.numperclass
		
        keylist = list(annotations.keys())
						
        random.shuffle(keylist)  
		
        selectedkeys = []
	
        annomatrix = np.zeros((len(keylist),3))
	
        annotationsSampled = []
	
        idx = 0
        for k in keylist:
            annomatrix[idx][0] = annotations[k][0] 
            annomatrix[idx][1] = annotations[k][1] 
            annomatrix[idx][2] = idx
           
            idx = idx + 1
    
        for c in popcls:
            annoc = annomatrix[annomatrix[:,0] == c]
			
            annotationsSampled.extend(annoc[:targetnum,2])
			
        for c in shotcls:
            annoc = annomatrix[annomatrix[:,1] == c]
			
            annotationsSampled.extend(annoc[:targetnum,2])
			
        popdict = {}
        shotclsdict = {}
		
        for id in annotationsSampled:
            id = int(id)
            popdict[keylist[id]] = annotations[keylist[id]][0]
            shotclsdict[keylist[id]] = annotations[keylist[id]][1]
			
        print('total nr images '+str(len(annotationsSampled)))
		
        writeCSV(args.outdir + '/popshotcls_sampled.csv',popdict,popdict,shotclsdict)
			
    elif args.task == 'samplemin':

        if len(args.annotationfile)>0:
            annotations = readCSV(args.annotationfile)
			
        elif len(args.annotationfilepattern)>0:
            files = glob.glob(args.annotationfilepattern)
		
            annotations = {}
		
            for filename in files:
                f = open(filename)
                annotationsf = readCSV(filename)
                annotations.update(annotationsf)
        

        excludelist = []
		
        if args.exclude is not None:
            excludedannotations = readCSV(args.exclude)
            excludelist = excludedannotations.keys()
		
        # classes to sample (excluding undefined, aerial)
        popcls = [0,1,2,3,4,5]
		
        if args.popcls is not None:
            popcls = np.array(args.popcls.split(',')).astype(np.int32)
		
        shotcls = [1,2,3,4,5,6,7,8,9]

        if args.shotcls is not None:
            shotcls = np.array(args.shotcls.split(',')).astype(np.int32)

		
        # build pools for annotations
        annotationsSampled = []
        popann = {}
        for v in popcls:
            popann[v] = []
        shotann = {}
        for v in shotcls:
            shotann[v] = []
		
        targetnum = args.numperclass
		
        keylist = list(annotations.keys())
				
        while not(checktargetnum(popann,shotann,targetnum,popcls,shotcls)<=0):
            		    		
            nextIdx = random.randint(0,len(keylist)-1)
			
            notForPopAnn = False
            notForShotAnn = False
			 
            if keylist[nextIdx] in excludelist:
                notForPopAnn = True
                notForShotAnn = True
 
			
            if annotations[keylist[nextIdx]][0] not in popcls:
                notForPopAnn = True
            else:
                notForPopAnn = len(popann[annotations[keylist[nextIdx]][0]])>=targetnum	
            if annotations[keylist[nextIdx]][1] not in shotcls:
                notForShotAnn = True
            else:
                notForShotAnn = len(shotann[annotations[keylist[nextIdx]][1]])>=targetnum	
			
            # special constraint to avoid same images for close-up and extreme close-up
            if not(notForShotAnn) and ("_ecu" in keylist[nextIdx]):
                if annotations[keylist[nextIdx]][1] != 1:
                    print('excluding '+keylist[nextIdx])
                    print(annotations[keylist[nextIdx]][1])
                    notForPopAnn = True
                    notForShotAnn = True
				
            while (nextIdx in annotationsSampled) or (notForPopAnn and notForShotAnn):
                nextIdx = random.randint(0,len(keylist)-1)
				
                notForPopAnn = False 
                notForShotAnn = False
                #print(nextIdx)
                #print(keylist[nextIdx])
                if annotations[keylist[nextIdx]][0] not in popcls:
                    notForPopAnn = True
                else:
                    notForPopAnn = len(popann[annotations[keylist[nextIdx]][0]])>=targetnum	
                if annotations[keylist[nextIdx]][1] not in shotcls:
                    notForShotAnn = True
                else:
                    notForShotAnn = len(shotann[annotations[keylist[nextIdx]][1]])>=targetnum		

                # special constraint to avoid same images for close-up and extreme close-up
                if not(notForShotAnn) and ("_ecu" in keylist[nextIdx]):
                    if annotations[keylist[nextIdx]][1] != 1:
                        print('excluding '+keylist[nextIdx])
                        print(annotations[keylist[nextIdx]][1])
                        notForPopAnn = True
                        notForShotAnn = True

	
            annotationsSampled.append(nextIdx)
			
            if not(notForPopAnn):
                popann[annotations[keylist[nextIdx]][0]].append(keylist[nextIdx])
            if not(notForShotAnn):
                shotann[annotations[keylist[nextIdx]][1]].append(keylist[nextIdx])
				
            if args.imgoutdir is not None:
                srcpath = args.imgbasepath
                if "_ecu" in keylist[nextIdx]:
                    srcpath = srcpath+'_ecu'
                shutil.copyfile(srcpath+'/'+keylist[nextIdx], args.imgoutdir+'/'+keylist[nextIdx])
			
			
        popdict = {}
        shotclsdict = {}
		
        for id in annotationsSampled:
            popdict[keylist[id]] = annotations[keylist[id]][0]
            shotclsdict[keylist[id]] = annotations[keylist[id]][1]
			
        print('total nr images '+str(len(annotationsSampled)))
		
        writeCSV(args.outdir + '/popshotcls_sampled.csv',popdict,popdict,shotclsdict)
	
    elif args.task == 'cvatimport':
	
        annotations = readCSV(args.annotationfile)
        fileidx = readJSONLines(args.imagelist,prefix='/')
		
        anno_json = []
        anno_obj = {}
        anno_obj['version'] = 3		
        anno_obj['shapes'] = []	
        anno_obj['tracks'] = []		
        anno_obj['tags'] = []
        anno_json.append(anno_obj)
		
        for a in annotations.keys():
            filekey = '/val_sample8'+a		
            idx = fileidx[filekey]
            annov = annotations[a]
            poplabel = getLabelName(annov[0],'pop')
            shotlabel = getLabelName(annov[1],'shot')
	
            tag = {}
            tag['frame'] = idx
            tag['group'] = 0
            tag['source'] = 'automatic'
            tag['attributes'] = []
            tag['label'] = poplabel
            anno_obj['tags'].append(tag)	

            tag = {}
            tag['frame'] = idx
            tag['group'] = 0
            tag['source'] = 'automatic'
            tag['attributes'] = []
            tag['label'] = shotlabel
            anno_obj['tags'].append(tag)	
			
				
        with open(args.outdir+'/annotations.json', 'w') as outfile:
            json.dump(anno_json, outfile)    

    elif args.task == 'cvat2csv':
			
        jsonf = f = open(args.annotationfile)
        data = json.load(jsonf)   
        fileidx = readJSONLines(args.imagelist,prefix='/')
		
        popdict = {}
        shotclsdict = {}		
		
        for tag in data[0]['tags']:
	
            fname = ''
            for fn in fileidx.keys():
                if fileidx[fn]==tag['frame']:
                    fname = fn
                    break
	
            poplabel = getLabelId(tag['label'],'pop')
            shotlabel = getLabelId(tag['label'],'shot')

            if poplabel !=999:
                popdict[fname] = poplabel
            if shotlabel != 999:
                shotclsdict[fname] = shotlabel
					
        writeCSV(args.outdir + '/popshotcls_cvat.csv',popdict,popdict,shotclsdict)

		
    elif args.task == 'timm':
        # classes to sample (excluding undefined, aerial)
        popcls = [0,1,2,3,4,5]
        shotcls = [1,2,3,4,5,6,7,8,9]

        annotations = readCSV(args.annotationfile)
        
        imgsourcedir = args.imgbasepath
        basedir = args.imgoutdir
        
        numperclass = args.numperclass
		 
        instancecount = {}
		
        if args.classtype=='pop':
            if args.binary_pop:
                label_names = ['unpopulated','populated']
                for c in [0,1]:
                    os.makedirs(basedir + '/'+ label_names[c].replace(' ','_'),exist_ok=True)
                    instancecount[c] = 0
            else:
                for c in popcls:
                    os.makedirs(basedir + '/'+ getLabelName(c,'pop').replace(' ','_'),exist_ok=True)
                    instancecount[c] = 0

        elif args.classtype=='shot':
            for c in shotcls:
                os.makedirs(basedir + '/'+ getLabelName(c,'shot').replace(' ','_'),exist_ok=True)
                instancecount[c] = 0

        annokeys = list(annotations.keys())
        random.shuffle(annokeys)   

        for a in annokeys:
            annov = annotations[a]
            if not(isinstance(annov, list)):
                annov = [ annov ]
            if args.classtype=='pop':
                if args.binary_pop:
                    if annov[0]<=2:
                        annov[0] = 0
                    else:
                        annov[0] = 1
                    mydir = label_names[annov[0]].replace(' ','_')	                    
                else:
                    mydir = getLabelName(annov[0],'pop').replace(' ','_')
                if annov[0]==999:
                    continue  
                if instancecount[annov[0]]>numperclass:
                    continue
                instancecount[annov[0]] += 1
            elif args.classtype=='shot':
                if annov[1]==0 or annov[1]==10:
                    continue  
                if instancecount[annov[1]]>numperclass:
                    continue
                mydir = getLabelName(annov[1],'shot').replace(' ','_')	            
                instancecount[annov[1]] += 1

            tokens = a.split('/')
            #trgname = tokens[-1]
            trgname = str(uuid.uuid4())+'.jpg'

            srcname = a
            thisimgsourcedir = imgsourcedir
            if 'ecu' in srcname:
                thisimgsourcedir = thisimgsourcedir + '_ecu/'

            if srcname[0] != '/':
                srcname = '/'+srcname
                trgname = '/'+trgname
            if os.path.exists(thisimgsourcedir+'/'+srcname):
                os.symlink(thisimgsourcedir+'/'+srcname,basedir+'/'+mydir+'/'+trgname)
            else:
                print(thisimgsourcedir+'/'+srcname + 'does not exist')

    elif args.task == 'eval':
        # classes to sample (excluding undefined, aerial)
        popcls = [0,1,2,3,4,5]
        shotcls = [1,2,3,4,5,6,7,8,9]

        annotations = readCSV(args.annotationfile)
        annotations2 = readCSV(args.annotationfile2)
	
        annokeys = list(annotations.keys())
        annokeys2 = list(annotations2.keys())
  
        popcorrect = 0
        popcount = 0
        shotcorrect = 0
        shotcount = 0
		 		
        for a in annokeys:
            if not a in annokeys2:
                continue
		
            annov = annotations[a]
            annov2 = annotations2[a]
			
            if annov[0] != 999:
                popcount += 1
                a1 = annov[0]
                a2 = annov2[0]
                if args.binary_pop:
                    if a1<1:
                        a1 = 0
                    else:
                        a1 = 1
                    if a2<1:
                        a2 = 0
                    else:
                        a2 = 1
                if a1 == a2:
                    popcorrect += 1
            if annov[1] != 0:
                shotcount += 1
                if args.acc_pm1:
                    if abs(annov[1] - annov2[1]) < 2:
                        shotcorrect += 1	
                else:
                    if annov[1] == annov2[1]:
                        shotcorrect += 1	
	
        print('accuracy population: '+str(popcorrect/popcount))
        print('accuracy shot type:  '+str(shotcorrect/shotcount))
		
				
    else:
        print('unknown task or task/parameter combination: '+args.task)
        exit()
	   
if __name__ == '__main__':
    
    main(sys.argv)
    

    

