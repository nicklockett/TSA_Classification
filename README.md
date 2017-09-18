# TSA_Classification
ECE 590-06 Classification Group

## Visualization Scripts
+ To run read_binary.py simply run the command `python read_binary.py [.aps or .a3d file] [output directory]` for example to run from the same directory as the .aps file and put the .mp4 file in the same place run `python2 read_binary.py fdb996a779e5d65d043eaa160ec2f09f.aps /`

## System Setup
+ download [python 2.7](https://www.python.org/downloads/) if it isn't already installed  
+ navigate to the top level of the git repo containing the `requirements.txt` file
+ `pip install -r requirements.txt` will give you all the python packages you need. NOTE if you have python 2 and 3 installed you may need to use `pip2`.

## System Requirements
python2.7

python packages
+ opencv
+ matplotlib
+ pandas
+ seaborn
+ numpy
+ scikit-learn
+ scipy

## Project Structure
    visualize/  
        classes.py # main script for running image split function and foot to head slice visualization
        read_binary.py # script from prof for turning the .a3d or .ads files into a rotating .mp4 video of the image
