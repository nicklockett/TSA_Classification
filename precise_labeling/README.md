# How to Label a Threat

## Quick Note
If you have issues with any of these steps, feel free to message in our slack, and we can give advice and/or offer suggestions! You may need to install a couple python packages if you don't already have them in order for this to run.

## Setup
1. Install ImageJ from https://imagej.nih.gov/ij/download.html 
2. Clone this git repo to your local machine
3. Install pypng by cloning the git repo from https://github.com/drj11/pypng.git and navigating to that project. Once you're within the project directory, run ```python setup.py install```. This can be done anywhere on your machine and does not need to be in the precise_labeling directory. You may need to run with ```--user``` if you have permission issues.
5. Download the .a3d files you will be using (outlined here: https://docs.google.com/spreadsheets/d/1x9bGCwdJ_ncP0VoQYyE9VbPjEzX-3OTnt9apPR-qBqw/edit#gid=0) into the TSA_Classification/precise_labeling/a3d/ directory local to this repo

## Label an Image
### Labeling Guidelines
* you must label each threat on any given image separately. Re-run the steps for each threat and differentiate the files using the region of the threat (file naming structure will be given below)
* Be sure to have the csv file with the threats open (in this directory) so that you can double check where each threat is and whether or not you are catching all of them (some of them are pretty difficult to see). But also keep in mind that some of their data is mislabeled, so if you really can't see the threat at all, then they may be wrong. Also keep the body_regions.png open.
* Be very careful to include all of the threat in your markings, with as little empty space as possible. It is essential to completely capture the threat. 

### Commands to Run
1. Make sure you are in the TSA_Classification/precise_labeling directory, and have already downloaded your a3d files.
2. run ``` python create_projection.py a3d/[filename] ``` this should output ```[filename]_projection.png``` in the projections/ dir
3. Open ```[filename]_projection.png``` in ImageJ (run ImageJ, a tool bar should come up, go to Open, then navigate to this directory and choose the image and it should be pulled up)
4. Draw a box around the threat (remember, we are labeling and saving one at a time)
5. click File -> Save As -> XY Coordinates
6. remove the 'projection' label at the end and add ```_xy_[region]``` to the end of the file so it follows the form 'fdb996a779e5d65d043eaa160ec2f09f_xy_3'. You may have to have the kaggle body zones image pulled up for this. save the file in the xyfiles/ dir.
7. run ``` python get_slice.py xyfiles/[filename]_xy_[region].txt``` this should output ```[filename]_xy_[region]_slice.png``` to the slices/dir
8. open ```slices/[filename]_xy_[region]_slice.png``` in ImageJ
9. draw a box around the threat
10. click File -> Save As -> XY Coordinates, save the file with the name unchanged in the xyfiles/ dir
11. run ``` python threat_localization.py [fileid] [region#]``` (Ex. ```python threat_localization.py 0043db5e8c819bffc15261b1f1ac5e42 1```) this should output ```[filename]_xy_[region]_threat.txt``` in the xyfiles/ dir. 
12. add all the final threat files to google drive here: https://drive.google.com/drive/folders/0B8ZE8AuY2VhrNFB0dmxFUTRoNVE?usp=sharing

### Troubleshooting
* if you have any trouble visually identifying a threat from the front facing projection we recommend making a slice with the process we've outlined and seeing if you can more clearly identify it with this slice. That has been effective for us. If you cannot ID a threat in a region that is said to have a threat do not submit one that you can't actually see.
* we recommend using python3 but when testing python27 worked fine as long as you don't have a 32 bit version
* if you have any issue with seaborn you may need to install numpy-1.13+mkl which can be found here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
