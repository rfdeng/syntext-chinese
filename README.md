# SynthText
This Code for generating synthetic text images is modified from https://github.com/ankush-me/SynthText
It can generate chinese sentences in horizontal or vertical directions:
![image](https://github.com/rfdeng/syntext-chinese/blob/master/samples_vertical.jpg)
![image](https://github.com/rfdeng/syntext-chinese/blob/master/samples_horizontal.jpg)


The corresponding paper is ["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

before running the code, the Pre-processed Background Images and the corresponding depth images, segmentation mask should be downloaded from the original website.  https://github.com/ankush-me/SynthText 

Dependencies and other preparation details can be found in the original website. It is recommended to read the original readme carefully before using this code.

To run the code:
python gen.py
