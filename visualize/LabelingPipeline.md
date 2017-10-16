### How to Label a Threat

## Setup
1. Install ImageJ from https://imagej.nih.gov/ij/download.html 
2. Download our python package into your directory housing your a3d files

## Label an Image
Some things to know
* you have to label each threat for an image separately. Re-run the steps for each threat and differentiate the files using the region of the threat
* Be sure to have to csv file with the threats open so that you can double check where each threat is and if you are catching all of them.  Just remember that some of those files are mislabled so don't take the information as cannon.
* Be very carefull to include all of the threat in your markings, with as little empty space as possible. It is essential to completely capture the threat. 

Commands to Run
1. run ``` python create_projection.py [filename] ``` this should output [filename]_projection.png
2. Open [filename]_projection.png in imageJ
3. Draw a box around the threat
4. click File -> Save As -> XY Coordinates
5. Add "_xy_[region]" to the end of the file
6. Save the file
7. run ``` python get_slice.py [filename]_xy_[region].txt``` this should output [filename]_xy_[region]_slice.png
8. open [filename]_xy_[region]_slice.png in ImageJ
9. Draw a box around the threat
10. click File -> Save As -> XY Coordinates
11. Save the file
12. run ``` python threat_localization.py [filename]_xy_[region].txt [filename]_xy_[region]_slice.txt``` this should output [filename]_xy_[region]_threat.txt
