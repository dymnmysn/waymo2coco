This repo is for converting Waymo dataset to Coco-Panoptic format for panoptic-segmentation.
Assuming the folder structure is as follows:
    -parent_dataset_folder
        -train
            -camera_image
                -57132587708734824_1020_000_1040_000.parquet
                -6128311556082453976_2520_000_2540_000.parquet
                ︙
            -camera_segmentation
                -57132587708734824_1020_000_1040_000.parquet
                -6128311556082453976_2520_000_2540_000.parquet
                ︙
        -val
            ︙
        -test
            ︙ 
        -waymo2coco
            -template.json
            -process.py
            -waymo2coco.py
------------------AFTER RUNNING THE CODE---------------------------
---------THE FOLLOWING FILES WILL BE GENERATED---------------------
        -formattedWaymo
            -train
                -0.jpg
                -1.jpg
                ︙
            -annotations
                -panoptic_train
                    -0.png
                    -1.png
                    ︙
                -panoptic_train.json
            -contextimmap_train.json

To convert the dataset you need to clone the repo to parent_folder.
Then by simply proceeding to waymo2coco folder and running process.py would be enough.

    conda create -n waymotococo python
    git clone ...git
    cd waymo2coco
    conda activate waymotococo
    pip install -r reqirements.txt
    python process.py

You can convert train or validation or test data. Train is default. For validation and test data you should run process.py with suitable arguments rather than default ones.