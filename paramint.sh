#!/bin/bash

python -c "import scipy.stats ; print(int(0.5+$1+($2-$1)*(scipy.stats.norm.cdf($3))))"
#python -c "print(int(max(1,64*(2**(${23})))))"
#python -c "print(int(max(1,64*(2**(${23})))))"
