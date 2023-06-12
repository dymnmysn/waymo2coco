import argparse
from waymo2coco import Waymo2Coco

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Waymo dataset to Coco format')
    parser.add_argument('--segm_parquet_dir', default='../train/camera_segmentation',help = 'waymo camera segmentation parquet files')
    parser.add_argument('--dataset_im_parquet_dir', default='../train/camera_image',help = 'waymo camera image parquet files')
    parser.add_argument('--savedir', default='../formattedWaymo/annotations/panoptic_train',help = 'Coco-Formatted segmentation images')
    parser.add_argument('--contextmappath', default='../formattedWaymo/contextimmap_train.json',help = 'Mapping between parquet and formatted files for debugging')
    parser.add_argument('--annfilename', default='../formattedWaymo/annotations/panoptic_train.json',help = 'Coco-Formatted panoptic annotations')
    parser.add_argument('--templatefilename', default='./template.json',help = 'Coco-annotation json format file')
    parser.add_argument('--camimgsavedir', default='../formattedWaymo/train',help = 'Training image path to extract jpg image to')
    parser.add_argument('--trainvaltest', default='train',help = 'Which data are you converting? Options are "train" or "val" or "test" or "all"')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    w2c = Waymo2Coco(args.segm_parquet_dir,args.dataset_im_parquet_dir,args.savedir,args.contextmappath,
                 args.annfilename,args.templatefilename,args.camimgsavedir,args.trainvaltest)
    w2c()

if __name__=='__main__':
    main()