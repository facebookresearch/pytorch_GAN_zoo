#!/bin/bash

python -c "import scipy.stats; import math ; print(math.exp(math.log($1)+(math.log($2)-math.log($1))*(scipy.stats.norm.cdf($3))))"
#python -c "print(int(max(1,64*(2**(${23})))))"
#python -c "print(int(max(1,64*(2**(${23})))))"
