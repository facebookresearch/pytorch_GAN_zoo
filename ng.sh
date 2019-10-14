#!/bin/bash 
python3.6 -u ng.py | stdbuf -o 0 -i 0 -e 0 tee nevergradpytorchganzoo_`date | sed 's/ /_/g'`_${1:-noinfo}
