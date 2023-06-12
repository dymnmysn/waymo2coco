This repo is for converting Waymo dataset to Coco-Panoptic format for panoptic-segmentation.\
Assuming the folder structure is as follows:\
____-parent_dataset_folder\
________-train\
________-camera_image\
____________-57132587708734824_1020_000_1040_000.parquet\
____________-6128311556082453976_2520_000_2540_000.parquet\
________________︙\
____________-camera_segmentation\
________________-57132587708734824_1020_000_1040_000.parquet\
________________-6128311556082453976_2520_000_2540_000.parquet\
________________︙\
________-val\
____________︙\
________-test\
____________︙ \
________-waymo2coco\
____________-template.json\
____________-process.py\
____________-waymo2coco.py\
------------------AFTER RUNNING THE CODE---------------------------\
---------THE FOLLOWING FILES WILL BE GENERATED---------------------\
________-formattedWaymo\
____________-train\
________________-0.jpg\
________________-1.jpg\
________________︙\
____________-annotations\
________________-panoptic_train\
____________________-0.png\
____________________-1.png\
____________________︙\
________________-panoptic_train.json\
____________-contextimmap_train.json\

To convert the dataset you need to clone the repo to parent_folder.\
Then by simply proceeding to waymo2coco folder and running process.py would be enough.\

    conda create -n waymotococo python\
    git clone ...git\
    cd waymo2coco\
    conda activate waymotococo\
    pip install -r reqirements.txt\
    python process.py\

You can convert train or validation or test data. Train is default. For validation and test data you should run process.py with suitable arguments rather than default ones.
