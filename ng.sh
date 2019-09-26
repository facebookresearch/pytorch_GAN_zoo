#!/bin/bash 
python3.6 -u ng.py | tee nevergradpytorchganzoo_`date | sed 's/ /_/g'`_${1:-noinfo}
