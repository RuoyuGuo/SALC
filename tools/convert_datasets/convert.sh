python cityscapes.py ../../../dataset/cityscapes/ --gt-dir gtFine -o ../../../dataset/cityscapes/

python gta.py ../../../dataset/gta/ --gt-dir labels -o ../../../dataset/gta/

python mapillary.py ../../../dataset/mapillary/half/ --gt-dir labels -o ../../../dataset/mapillary/half/

python synthia.py ../../../dataset/RAND_CITYSCAPES/ --gt-dir GT/LABELS -o ../../../dataset/RAND_CITYSCAPES/