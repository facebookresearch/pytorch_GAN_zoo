num=`ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*/*.jpg | sed 's/.*rand_//g' | sed 's/_.*//g' |  sort -n | tail -n 1`

i=0
rm compare*.jpg
echo target
ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*/*.jpg | grep -iv celeb |grep "rand_${i}_" | sed 's/.*_rand//g' | sed 's/.jpg//' | sed 's/nsteps.*//g' | sed 's/random_search/rs/g' | sed 's/.*_//g'
for i in `seq 0 $num`
do
echo "image $i:" `ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*/*.jpg | grep -iv celeb | grep "rand_${i}_" | sed 's/.*_rand//g' | sed 's/.jpg//' | sed 's/nsteps.*//g' | sed 's/random_search/rs/g' | sed 's/.*_//g'`
#    ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*.jpg | grep "rand_${i}\."
 #   ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*/*.jpg | grep "rand_${i}_" | wc -l
    num2=`ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*/*.jpg | grep "rand_${i}_" | grep -iv celeb | wc -l`
    montage `ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*.jpg | grep -iv celeb | grep "rand_${i}\."` `ls -ctr /private/home/oteytaud/HDGANSamples/random_gens/*/*.jpg | grep -iv celeb |grep "rand_${i}_"` -tile $((${num2} + 1))x1 -geometry +0+0 compare${i}.jpg

done
readlink -f compare*
montage compare*.jpg -tile 1x$(( $num + 1)) -geometry +0+0 overview.jpg
readlink -f overview.jpg
grep ptimal `ls -ctr resul* | tail -n 1` | sed 's/nsteps.*losses..//g' | sed 's/.*L2_//g' | sed 's/\\n.*//g' 
grep ptimal `ls -ctr resul* | tail -n 1` | sed 's/nsteps.*losses..//g' | sed 's/.*L2_//g' | sed 's/\\n.*//g' | awk '{seen[$1]+=$2; count[$1]++} END{for (x in seen)print x, seen[x]/count[x]}' 

