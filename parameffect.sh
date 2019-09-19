#!/bin/bash

rm plotparams/*.png


(echo -n "FIT "
echo -n "NS MSF P5555 ASB "
echo -n "P2222 P1111 RPM NH "
echo -n "CG WD P444 UB "
echo -n "ZBGSZ ADAM NL MOM "
echo -n "FB TH B2 LR "
echo -n "GSTM BLS LMBDA P88 "
echo    "TRB P222 WN "
for i in `grep -c "\-ns 1" output*.out | grep ":0" | sed 's/:.*//g'`
do
# echo "#$i"
echo -n `grep __log__ $i  | sed 's/.*bpd..//g'  | grep -v NaN |sed 's/,.*//g'|grep "[0-9]" | awk 'BEGIN{a=10000000}{if ($1<0+a) a=$1} END{print a}' ` "   "
grep "params=" $i | sed 's/params=//g'
done | grep -v "^1000" ) > /tmp/dataparams.txt

numfields=`head -n 1 /tmp/dataparams.txt | awk '{print NF }'`
echo $numfields
for i in `seq 2 $numfields`
do
 ipo=$(( $i + 1 ))
 for j in `seq $ipo $numfields`
 do
  str="{ print \$$i, \$$j, \$1 }"
  echo $str
  awk "$str" /tmp/dataparams.txt | head -n 3
  awk "$str" /tmp/dataparams.txt | python3 plot2params.py 
  sleep 0.1
 done
done
for i in `seq 2 $numfields`
do
  str="{ print \$$i, \$1 }"
  echo $str
  awk "$str" /tmp/dataparams.txt | head -n 3
  awk "$str" /tmp/dataparams.txt | python3 plotparam.py 
  sleep 0.1
done
