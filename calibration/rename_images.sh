#!/bin/bash
cd ~/projects/calibration/images

k = 0
for i in *
do 
    mv "${i}" image$((++k)).jpg
done

cd ~/projects/calibration
