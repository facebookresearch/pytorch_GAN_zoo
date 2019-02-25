#!/bin/bash

for full in restricted full
do
for nstep in 200 1000 5000
do
for inspire in dtd20 celeba dtd20miss celebacartoon2
do
for renorm in renorm none
do
for loss in l2 vgg mixed closs dloss
do
./launch_celeba_camille.sh $inspire $full $loss $renorm $nstep
done
done
done
done
done

