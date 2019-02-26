#!/bin/bash

storagedir="~/storage`date | sed 's/ /_/g'`"
mkdir -p $storagedir
mv overview*.jpg list*.txt $storagedir
#for full in full
for full in restricted 
do
for nstep in 200 1000 5000
do
for inspire in celebacartoon2
#for inspire in dtd20 celeba dtd20miss celebacartoon2
do
for renorm in renorm none
do
for loss in l2 vgg mixed closs dloss
do
./launch_celeba_camille.sh $inspire $full $loss $renorm $nstep
sleep 1
done
done
done
done
done

