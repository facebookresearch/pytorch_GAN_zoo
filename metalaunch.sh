#!/bin/bash

storagedir="~/storage`date | sed 's/ /_/g'`"
mkdir -p $storagedir
mv overview*.jpg list*.txt $storagedir
#for full in full
for nstep in 200 1000 5000
do
for full in full
#for full in restricted full
do
for inspire in celebabam   #udtd20 dtd20 dtd20miss celeba
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

