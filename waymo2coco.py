import dask.dataframe as dd
from PIL import Image
import numpy as np
import json
import io
import pathlib
from itertools import repeat
from multiprocessing import Pool
import os

class Waymo2Coco:
    def __init__(self,segm_parquet_dir,dataset_im_parquet_dir,savedir,contextmappath,
                 annfilename,templatefilename,camimgsavedir,trainvaltest):
        self.segm_parquet_dir = segm_parquet_dir
        self.dataset_im_parquet_dir = dataset_im_parquet_dir
        self.savedir = savedir
        self.contextmappath = contextmappath
        self.annfilename = annfilename
        self.templatefilename = templatefilename
        self.camimgsavedir = camimgsavedir
        self.trainvaltest = trainvaltest
        self.startnum = 0
        self.startnumtrain = 0
        self.startnumval = 70000
        self.startnumtest = 90000
        self.mydict = None
        self.mydictT = None
        self.myvalues = None

    @staticmethod
    def getdatafromcontext(c_segm_path):
        c = dd.read_parquet(c_segm_path)
        contextimages = []
        for row in c.iterrows():
            data = row[1]
            con = data[0]
            time = data[1]
            cam = data[2]
            pan = data[4]
            contextimages.append((con,time,cam,pan))
        return contextimages

    def processcontexts(self,context_path_list):
        with Pool() as p:
            allimages = p.map(Waymo2Coco.getdatafromcontext,context_path_list)
        return allimages

    def saveallcontextsmapping(self):
        allcontexts_not_flatten = self.processcontexts(self.contexts)
        allcontexts = []
        for h in allcontexts_not_flatten:
            allcontexts+=h
        allmapping = {i: (a,b,c) for i,(a,b,c,_) in enumerate(allcontexts,start = self.startnum)}
        os.makedirs(os.path.dirname(self.contextmappath), exist_ok=True)
        with open(self.contextmappath,'w') as f:
            json.dump(allmapping,f)
        return allcontexts

    @staticmethod
    def getobjidmap(im_order, s):
        return {el : im_order + 111847*k for k,el in enumerate(s)} #assuming maximum image order < 111847
    
    @staticmethod
    def settoobjids(pix,map):
        return map[pix]

    @staticmethod
    def convertim(sinput):
        thingclasses = {3: 'CAR', 4: 'TRUCK', 5: 'BUS', 6: 'OTHER_LARGE_VEHICLE', 9: 'TRAILER', 10: 'PEDESTRIAN', 11: 'CYCLIST', 12: 'MOTORCYCLIST'}
        (imorder, (_,date_captured,_,cntxtim)), savedir = sinput
        i = Image.open(io.BytesIO(cntxtim))
        width,height = i.size
        file_namejpg= f'{imorder}.jpg'
        imageentry = dict(file_name=file_namejpg, height=height, width = width, date_captured=date_captured, id=imorder)
        img = np.asarray(i)
        shape = img.shape
        img1 = img.transpose()
        flatimage = img.flatten()
        segset = set(flatimage)
        objmap = Waymo2Coco.getobjidmap(imorder, segset)
        items = []
        file_name = f'{imorder}.png'
        image_id = imorder

        objids = [Waymo2Coco.settoobjids(pix,objmap) for pix in flatimage]
        rgb = np.reshape(objids,shape)
        blue = rgb//65536
        green = (rgb%65536)//256
        red = rgb%256 
        nimg = np.stack([red,green,blue], axis=-1)
        impathsave = f'{savedir}/{imorder}.png'
        os.makedirs(os.path.dirname(impathsave), exist_ok=True)
        Image.fromarray(nimg.astype('B')).save(impathsave)

        for s in segset:
            if int(s//1000 + 1) in thingclasses.keys():
                item = dict()
                item['id'] = objmap[s]
                item['category_id'] = int(s//1000 + 1)
                item['iscrowd'] = 0
                mask = np.where(img1==s)
                xmin, xmax, ymin, ymax, area = mask[0].min(), mask[0].max(), mask[1].min(), mask[1].max(), len(mask[0])
                bbox = [int(xmin), int(ymin), int(xmax-xmin), int(ymax - ymin)]
                item['bbox'] = bbox
                item['area'] = int(area)
                items.append(item)
        annotationentry = dict(segments_info = items, file_name = file_name, image_id = image_id)
        return imageentry,annotationentry
    
    def savedict(self):
        with open(self.contextmappath,'r') as f:
            self.mydict = json.load(f)

        self.myvalues = set(tuple(value) for _,value in self.mydict.items())
        self.mydictT = {tuple(value):key for key,value in self.mydict.items()}
        print('Generating and saving annotation json and png files...',end='\t: ')
        with Pool() as p:
            allobjectmaps = p.map(Waymo2Coco.convertim,zip(self.allcontextsindexed,repeat(self.savedir)))
        print('Completed.')
        with open(self.templatefilename,'r') as f:
            template = json.load(f)

        template['images'] = [aom[0] for aom in allobjectmaps]
        template['annotations'] = [aom[1] for aom in allobjectmaps]
        os.makedirs(os.path.dirname(self.annfilename), exist_ok=True)
        with open(self.annfilename,'w') as f:
            json.dump(template, f)

    @staticmethod
    def saveimagesfromcontext(tinput):
        contextpath, mydictT, myvalues, camimgsavedir = tinput
        c = dd.read_parquet(contextpath)
        for row in c.iterrows():
            data = row[1]
            contimecam = (data[0],data[1],data[2])
            if contimecam in myvalues:
                jimg = data[3]
                imorder = mydictT[contimecam]
                imgfile = f'{camimgsavedir}/{imorder}.jpg'
                os.makedirs(os.path.dirname(imgfile), exist_ok=True)
                with open(imgfile, "wb") as binary_file:
                    binary_file.write(jimg)

    def changedefaulttraintoval(self):
        self.segm_parquet_dir = self.segm_parquet_dir.replace('train','val')
        self.dataset_im_parquet_dir = self.dataset_im_parquet_dir.replace('train','val')
        self.savedir = self.savedir.replace('train','val')
        self.contextmappath = self.contextmappath.replace('train','val')
        self.annfilename = self.annfilename.replace('train','val')
        self.templatefilename = self.templatefilename.replace('train','val')
        self.camimgsavedir = self.camimgsavedir.replace('train','val')
        self.trainvaltest = self.trainvaltest.replace('train','val')
        self.startnum = self.startnumval
    
    def changedefaulttraintotest(self):
        self.segm_parquet_dir = self.segm_parquet_dir.replace('train','test')
        self.dataset_im_parquet_dir = self.dataset_im_parquet_dir.replace('train','test')
        self.savedir = self.savedir.replace('train','test')
        self.contextmappath = self.contextmappath.replace('train','test')
        self.annfilename = self.annfilename.replace('train','test')
        self.templatefilename = self.templatefilename.replace('train','test')
        self.camimgsavedir = self.camimgsavedir.replace('train','test')
        self.trainvaltest = self.trainvaltest.replace('train','test')
        self.startnum = self.startnumtest
    
    def changedefaultvaltotest(self):
        self.segm_parquet_dir = self.segm_parquet_dir.replace('val','test')
        self.dataset_im_parquet_dir = self.dataset_im_parquet_dir.replace('val','test')
        self.savedir = self.savedir.replace('val','test')
        self.contextmappath = self.contextmappath.replace('val','test')
        self.annfilename = self.annfilename.replace('val','test')
        self.templatefilename = self.templatefilename.replace('val','test')
        self.camimgsavedir = self.camimgsavedir.replace('val','test')
        self.trainvaltest = self.trainvaltest.replace('val','test')
        self.startnum = self.startnumtest

    def call(self):
        self.contexts = list(pathlib.Path(self.segm_parquet_dir).glob('*.parquet'))
        self.contexts_cam_images = list(pathlib.Path(self.dataset_im_parquet_dir).glob('*.parquet'))
        self.allcontexts = self.saveallcontextsmapping()
        self.allcontextsindexed = list(enumerate(self.allcontexts, start = self.startnum))
        self.savedict()
        print('Saving image data...',end='\t: ')
        with Pool() as p:
            p.map(Waymo2Coco.saveimagesfromcontext, zip(self.contexts_cam_images,
                    repeat(self.mydictT),repeat(self.myvalues),repeat(self.camimgsavedir)))
        print('Completed.')
            
    def __call__(self):
        if self.trainvaltest == 'train':
            print("Conversion of train data started")
            self.call()
            print("Conversion of train data completed")
        elif self.trainvaltest == 'val':
            self.changedefaulttraintoval()
            print("Conversion of val data started")
            self.call()
            print("Conversion of val data completed")
        elif self.trainvaltest == 'test':
            self.changedefaulttraintotest()
            print("Conversion of val data started")
            self.call()
            print("Conversion of test data completed")
        elif self.trainvaltest == 'all':
            print("Conversion of train data started")
            self.call()
            print("Conversion of train data completed")
            self.changedefaulttraintoval()
            print("Conversion of val data started")
            self.call()
            print("Conversion of val data completed")
            self.changedefaultvaltotest()
            print("Conversion of test data started")
            self.call()
            print("Conversion of test data completed")
