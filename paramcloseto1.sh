#!/bin/bash

python -c "import scipy.stats; import math ; print(1.-math.exp(math.log(1-$1)+(math.log(1-$2)-math.log(1-$1))*(scipy.stats.norm.cdf($3))))"
#python -c "print(int(max(1,64*(2**(${23})))))"
#python -c "print(int(max(1,64*(2**(${23})))))"
