
username=`whoami`
num=`ls -ctr /private/home/${username}/HDGANSamples/random_gens/*/*.jpg | sed 's/.*rand_//g' | sed 's/_.*//g' |  sort -n | tail -n 1`

i=0
rm compare*.jpg
for i in `seq 0 $num`
do
echo "image $i:" `ls -ctr /private/home/${username}/HDGANSamples/random_gens/*/*.jpg |  grep "rand_${i}_" | sed 's/.*_rand//g' | sed 's/.jpg//' | sed 's/nsteps.*//g' | sed 's/random_search/rs/g' | sed 's/.*_//g'`
    num2=`ls -ctr /private/home/${username}/HDGANSamples/random_gens/*/*.jpg | grep "rand_${i}_" |  wc -l`
    convert -resize 128x128 `ls -ctr /private/home/${username}/HDGANSamples/random_gens/*.jpg |  grep "rand_${i}\."` to.jpg
    montage to.jpg `ls -ctr /private/home/${username}/HDGANSamples/random_gens/*/*.jpg | grep "rand_${i}_"` -tile $((${num2} + 1))x1 -geometry +0+0 compare${i}.jpg

done
echo 'list of scores for each image and each method --------------------------- (lower=better)'
grep 'noptimal.losses' `ls -ctr rescelebaul* | tail -n 1` |sed 's/nsteps.*losses..//g' | sed 's/.*L2_//g' | sed 's/\\n.*//g' 
(
echo 'list of average score for each method --------------------------- (lower=better)'
grep 'noptimal.losses' `ls -ctr rescelebaul* | tail -n 1` |sed 's/nsteps.*losses..//g' | sed 's/.*L2_//g' | sed 's/\\n.*//g' | awk '{seen[$1]+=$2; count[$1]++} END{for (x in seen)print x, seen[x]/count[x]}' ) > listscores.txt
(echo 'list of cols for the final overview of inspired images --------------------'
echo '(all cols except the first refer to rebuilt images)'
echo target
ls -ctr /private/home/${username}/HDGANSamples/random_gens/*/*.jpg | grep "rand_${i}_" | sed 's/.*_rand//g' | sed 's/.jpg//' | sed 's/nsteps.*//g' | sed 's/random_search/rs/g' | sed 's/gradient_descent/gs/g' | sed 's/.*_//g' ) > listcols.txt
echo '========================================================='
echo '========================================================='
echo '========================================================='
echo '========================================================='
echo '========================================================='
cat listcols.txt
cat listscores.txt
suffix=${inspire}_${nstep}_${renorm}_${full}_${loss}
cp listscores.txt listscores_${suffix}.txt
montage compare*.jpg -tile 1x$(( $num + 1)) -geometry +0+0 overview${suffix}.jpg
tar -zcvf ~/overview${suffix}.tgz listcols.txt listscores.txt overview${suffix}.jpg
echo 'image with target and rebuilt images -------------------------------'
readlink -f overview${suffix}.jpg | sed 's/^/~\/teleview.sh /g'
echo everything you need: `readlink -f ~/overview${suffix}.tgz`
mkdir -p ~/overviews
touch ~/overviews/overover.jpg
rm ~/overviews/over*.jpg
cp overview*.jpg ~/overviews/
