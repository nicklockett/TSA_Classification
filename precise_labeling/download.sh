#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
	set timeout -1
	scp rec
	#python create_projection.py a3d/$line
	#python get_slice.py xyfiles/$line
    #python threat_localization.py $line
done < "$1"