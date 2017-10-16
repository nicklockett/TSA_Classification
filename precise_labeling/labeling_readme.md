### How to Label a Threat

## Setup
1. Install ImageJ from https://imagej.nih.gov/ij/download.html 
2. Download our python package into your directory housing your a3d files

## Label an Image
Some things to know
* you have to label each threat for an image separately. Re-run the steps for each threat and differentiate the files using the region of the threat
* Be sure to have to csv file with the threats open so that you can double check where each threat is and if you are catching all of them.  Just remember that some of those files are mislabeled so don't take the information as cannon.
* Be very careful to include all of the threat in your markings, with as little empty space as possible. It is essential to completely capture the threat. 

Commands to Run
1. run ``` python create_projection.py a3d/[filename] ``` this should output [filename]_projection.png in the projections/ dir
2. Open [filename]_projection.png in ImageJ
3. Draw a box around the threat
4. click File -> Save As -> XY Coordinates
5. add "_xy_[region]" to the end of the file so it follows the form 'fdb996a779e5d65d043eaa160ec2f09f_xy_3'
6. save the file in the xyfiles/ dir
7. run ``` python get_slice.py xyfiles/[filename]_xy_[region].txt``` this should output [filename]_xy_[region]_slice.png to the slices/dir
8. open slices/[filename]_xy_[region]_slice.png in ImageJ
9. draw a box around the threat
10. click File -> Save As -> XY Coordinates
11. save the file with the name unchanged in the xyfiles/ dir
12. run ``` python threat_localization.py xyfiles/[filename]_xy_[region].txt xyfiles/[filename]_xy_[region]_slice.txt``` this should output [filename]_xy_[region]_threat.txt in the xyfiles/ dir
13. add all threatcube files to the server in ___ location
