**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: left.png "Left Image"
[image2]: right.png "Right Image"
[image3]: center.png "Normal Image"
[image4]: flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py/ lab.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-10.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model-10.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 184-188) 

The model includes RELU layers to introduce nonlinearity (code line 184-188), and the data is normalized in the model using a Keras lambda layer (code line 182). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 195).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I use the same network which is used at NVIDIA since its already prooven. I did think of adding certain dropout layers to it, but it worked fine with a little preprocessing.

For preprocessing I  -
- Flipped the images
- Added shadow to the images 
- Changed brightness of the images
- In my keras layer i also removed the top and bottom portions of the images since they were redundant.(showing car & peaks & trees)

In the case of left and right images I added a correction of 0.1 which gave me best results. I had to play with this correction factor alot before getting the right answers

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes -
 - Normalizing layer
 - 3 CNN layers with 5x5 filters
 - 2 CNN layers with 2x2 filters
 - 1 Flattening layer
 - 4 Fully connected layers with last layer of size 1 since this is a regression case.






####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center incase it drifts off

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 although the difference is not much greater in cast of it being 5 I used an adam optimizer so that manually training the learning rate wasn't necessary.
