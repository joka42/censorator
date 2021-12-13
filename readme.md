# Censorator
The censorator program detects nudity in images and censores them. 

## Installation and Usage
### Dependencies in Linux / WSL
Install python version 3.6 or later on your system. If you are using Linux it should alread be installed on your system.
However, if it is not, you can install it using your package manager, e.g.:

    sudo apt install python3

In order to censor animated gif images you need to install ffmpeg as well:

    sudo apt install ffmpeg

### Dependencies Windows
I do not test on windows, but you can try to run the script in windows as well. I would strongly recommend to use either
linux or activate and use the Windows Subsystem for Linux (WSL) (https://lmgtfy.app/?q=how+to+install+wsl2+on+windows).
Then you can follow the instructions for Linux/WSL

* Install python3 using a guide of your choice (https://lmgtfy.app/?q=how+to+install+python3+in+windows)
* Install ffmpeg from a source of your choice (https://lmgtfy.app/?q=ffmpeg+for+windows+download) and follow the steps 
under usage.

## Usage
You need to install python dependencies. It is best practice to NOT install these dependencies system wide, but into a
virtual environment (venv) for python. How to do this? 

Clone or download the repository into a folder on your system, enter that directory in your terminal/cmd

    cd path/to/censorator

Create a virtual environment with python:

    python3 -m venv venv

Activate the environment:

    source venv/bin/activate

Install all dependencies for the script:

    pip install -r requirements.txt

Now you should be able to run the script using

    python main.py -i path/to/image

### Advanced Command Line Options

    -i / --input:   input path. Image file or directory. If it is a directory, all images in the directory will be processed.

    -o / --output:  [optional] Output directory. Censored images will be stored in this directory. If no output directory is specified, the working directory/current directory is used

    -s / --strict:   [optional] Censor also covered breasts and butts. 

    -c / --casual:   [optional] Only censor exposed breasts and female genitalia

    --stamped:       [optinal] Apply the pussy-free stamp to female genitalia

    --skip-existing: [optinal] Skip images that are already in the output directory, do not overwrite them.

    --filter:        [optional] Apply a (Kalman-like) filter to the results for animated images. This might help to censor frames in which the neuronal net did not identify all areas of interest.

    --debug:         [optoinal] Output all debug information and apply a dot in the center of identified areas (should only be used for debugging, because the output images contain the red dots).



