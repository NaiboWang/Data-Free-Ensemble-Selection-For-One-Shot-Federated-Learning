#!/usr/bin/bash

NAME="femnist"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME