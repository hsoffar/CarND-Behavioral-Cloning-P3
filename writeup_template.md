# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./network_architecture.png "Model Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The Nivida Model is used as a startign point.

![alt text][image1]
#### 2. Attempts to reduce overfitting in the model

In an effort to reduce overfitting and increase my model's ability to generalize for driving on unseen roads, I artificially increased my dataset using a couple of proven image augmentation techniques. 
- adjusting the brightness of the images. 
- scaling up or down the V channel by a random factor
- crop the top 40 pixels and the bottom 20 pixels from each image in order to remove any noise from the sky or trees in the top of the images and the car's hood from the bottom of the image.

### 3. Data Collection
I used only the sample data provided by Udacity to train my model

#### 4. Appropriate training data

The data provided by Udacity contains only steering angles for the center image, so in order to effectively use the left and right images during training, I added an offset of .275 to the left images and subtracted .275 from the right images. 
Applying the steps mentioned in the "attempts to reduce overfiting"

The model was trained on:

Hardware:
Processor: Intel i5
Graphics card: GeForce GTX 700

the model was run on a GPU

The entire set of images used for training would consume a large amount of memory. A python generator is used so that only a single batch is contained in memory at a time.

### using video.py
python video.py run1
Creates a video based on images found in the run1 directory. The name of the video will be the name of the directory followed by '.mp4', so, in this case the video will be run1.mp4.

Optionally, one can specify the FPS (frames per second) of the video:

python video.py run1 --fps 48
Will run the video at 48 FPS. The default FPS is 60.

## Please run1.mp4

### Recources
* Python Environment: CarND-Term1-Starter-Kit
* Nvidia paper: End to End Learning for Self-Driving Cars
* Project specification: Udacity Rubrics
* Udacity repository: CarND-Behavioral-Cloning-P3
