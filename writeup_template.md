
[//]: # (Image References)

[model1]: ./images/20170320.01-conv_4_fc_3-b32-e30-d0.63_0.63-s0.1-v0.2-z0.8-model.png "Winning Model Visualization"
[histogram_all]: ./images/histogram_all_angles.png "Histogram of all angles"
[histogram_ignored80]: ./images/histogram_ignore80.png "Histogram of angles ignoring 80% of zero angles"
[center_image]: ./images/center.jpg "Center image"
[left_image]: ./images/left.jpg "Left image"
[right_image]: ./images/right.jpg "Right image"
[normal_image]: ./images/center.jpg "Normal image"
[flipped_image]: ./images/flipped.jpg "Flipped image"

[//]: # (Link References)

[model.py]: ./model.py "model defining file"
[drive.py]: ./drive.py "simulator driving file"
[project3.py]: ./project3.py "wrapper script for model.py"
[train_the_ocean.sh]: ./train_the_ocean.sh "shell script to run many, many experiments"
[video.mp4]: ./video.mp4 "video of the model driving around track one"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following standard files:

- **[model.py]** - contains the script to create and train the model
- **[drive.py]** - drives the car in autonomous mode
- **model.h5** - contains a trained convolution neural network 
- **writeup_report.md** - this file summarizing the results
- **[video.mp4]** - a video of my best model driving the car autonomously

Additionally, specific to this project, I have the following important files:

- **[project3.py]** - a wrapper script that takes hyper parameters as input and trains a model
- **[train_the_ocean.sh]** - an example bash script used to run many experiments
- **[videos/*](./videos/)** - all videos of the successful models

#### 2. Submission includes functional code

My [project3.py] wrapper script can be run with default parameters:
```sh
python project3.py
```

This will run the default model (called *conv_4_fc_3*) on all input data found in *data/\*/driving_log.csv* and saving the results to *results/*.

The accepted parameters are:

- **-m** name of the model to use
- **-e** max number of epochs to run
- **-b** batch size
- **-d** list of dropout values to use
- **-s** list of sample set directories to use for training
- **-i** extra identifier to include in the output file names
- **-v** percentage of sample set to use for validation
- **-x** absolute value of steering correction to use for left and right cameras
- **-z** percentage of samples with steering angle of 0.0 to ignore
- **-c** only use center camera images

Using the Udacity provided simulator and my [drive.py] file, the car can be driven autonomously around the track by executing:

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py] file contains functions for reading in the sample data and creating 5 different network architectures. Additionally, it has example code for training and saving the trained models, but in practice, this is generally done in the [project3.py] wrapper script to facilitate experimentation. 

This contains multiple networks, because I like to experiment with different architectures, sometimes simultaneously. It also provides a log of what I've tried when approaching future problems.

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My *winning* model, called [conv_4_fc_3](./model.py#L107L169), consists of 4 convolution layers followed by 3 fully connected layers. By **winning** I mean the best model among the models I tried.

The first two convolution layers use a 5x5 filter size and have depths of 8 and 16. The last two convolution layers use a 3x3 filter size and have depths of 32 ([model.py, lines 133 - 151][./model.py#L133L151]).

Prior to the first convolution layer, the model includes a Keras cropping layer to take off the top 56 and bottom 24 rows of pixels off the input ([model.py, line 128](./model.py#L128)). It then mean centers the pixels using a Keras lambda layer ([model.py, line 131](./model.py#L131)).

The model uses RELU activation layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers for the first two fully connected layers in order to reduce overfitting ([model.py, line 158](./model.py#L158) and [model.py, line 163](./model.py#L163)). It does not make sense to use dropout on the final fully connected layer, as then the model will just output a zero during passes when dropout is in effect.

The model was trained and validated on several different data sets to ensure that the model was not overfitting. This is not hard coded into the file, but rather passed as a command line argument (-s) to the [project.py] wrapper script.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The hyper parameters of my final model are:

- batch size: **32**
- epochs: **30** (although early stopping and model checkpoints were implemented)
- dropout: **0.63** for the first and second fully connected layers

The model used an adam optimizer, so the learning rate was not tuned manually ([project3.py, line 107](./project3.py#L107)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a lot of training samples (upwards of 130,000) from multiple runs around the track in both directions. This gave the model a large variety of inputs to learn from. These training samples incorporated the center, left, and right camera angles from the recorded laps.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I **really** like to experiment. I tend to experiment a lot, especially in subjects that interest me. In arriving at my solution, I played with 5 different model architectures and ran over 100 experiments across them.

The models I tried out include:

- **[conv_4_fc_3](./model.py#L107L169)**: My eventual best model comprising of 4 convolution layers and 3 fully connected layers.
- **[conv_4_fc_3_more_filters](./model.py#L41L104)**: A modification of *conv_4_fc_3* that added significantly more filters to the convolution layers
- **[resnet_ish](./model.py#L172L226)**: An attempt to try transfer learning by using a ResNet50 model pretrained on ImageNet with new fully connected layers attached to the end of the convolution layer stack
- **[vgg16_ish](./model.py#L229L283)**: Another attempt to try transfer learning by using a VGG16 model pretrained on ImageNet
- **[end_to_end_nvidia](./model.py#L286L362)**

My experimenting cycle:

1. Deciding on 6-12 experiments to run
2. Firing up a p2.xlarge spot instance (**significantly** cheaper as a spot instance!)
3. Updating [train_the_ocean.sh] with my experiment parameters
4. Running the experiments (usually over night)
5. Downloading the results
6. Shutting down the spot instance
7. Viewing the results in autonomous mode of the simulator
8. Based on results, goto 1.

Relatively quickly, I had to rule out further experiments with **resnet_ish** and **vgg16_ish**. The problem I had was that my laptop was too slow to run the models fast enough for the simulator. Instead of trying to figure out if I could run the simulator on the p2.xlarge spot instance, I decided to discard those models in the interest of time.

My winning[^winning] mode was **conv_4_fc_3**, ironically, the first model I created. I started with the final model from my [Traffic Sign Recognizer Project](https://github.com/yonomitt/traffic-sign-classifier) and modified it. I added a convolution layer tweaked the number of filters slightly.

I created **conv_4_fc_3_more_filters** because I was interested in seeing the effect of vastly increasing the number of filters of each convolution layer on the final results. Quite to my surprise, it performed worse, though this could have something to do with the size of the fully connected layers at the end.

I also knew from the beginning that I wanted to try out the End to End NVIDIA model as outlined in [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). As I had to tweak it slighty due to different input sizes, I think I probably made some inoptimal decisions. The result of the model was rather disappointing, so I'm sure I made some mistakes along the way. However, since I was running out of time and another model was working, I didn't investigate this further.

#### Early Stopping

I used Keras's early stopping callback to stop training the model if the validation loss did not improve in the last 4 epochs. This is intended to help prevent overfitting.

#### Model Checkpoint

I made use of Keras model checkpoint callback to save the model at the end of each epoch provided that the validation loss had decreased. While this could potentially increase the chance of overfitting, as the saved model is kind of also training on the validation set, I used another method to combat overfitting.

#### Dropout

I implemented dropout on all of my fully connected layers (besides the output layer) for each of my models. I discovered during my [Traffic Sign Recognizer Project](https://github.com/yonomitt/traffic-sign-classifier), that this was sufficient and normally worked better for me than having dropout on the convolutional layers.

An interesting correlation I seemed to notice between dropout and the autonomous simulator driving is that the higher the dropout, the smoother the drive. Frequently with a lower dropout rate, the models would drive the car in a sinusoidal pattern between the edges of the lanes (bouncing back and forth). The models that ran with a higher dropout rate were significantly smoother and reduced this sinusoidal motion.

After all of my experiments, I was able to get 4 models to drive all the way around the track without leaving the road. From best to worst:

1. **conv_4_fc_3** - winner
	- dropout: 0.63 for both fully connected layers
	- steering correction: 0.1
	- zeros angle inputs ignored: 80%
2. **conv_4_fc_3** - left the lane once but not the road
	- also cut left and right camera inputs if the center steering angle was 0.0
	- dropout: 0.5 for both fully connected layers
	- steering correction: 0.1
	- zeros angle inputs ignored: 80%
3. **conv_4_fc_3** - left the lane twice but not the road
	- dropout: 0.5 for both fully connected layers
	- steering correction: 0.1
	- zeros angle inputs ignored: 80%
4. **conv_4_fc_3_more_filters** - drunk driving
	- dropout: 0.37 for both fully connected layers
	- steering correction: 0.1
	- zeros angle inputs ignored: 80%

As you can see all the models that completed the track are fairly similar with usually just a difference in the dropout rate.

#### 2. Final Model Architecture

The final model architecture consisted of:

- Convolutional layer with a 5x5 filter size and a depth of 8
- Convolutional layer with a 5x5 filter size and a depth of 16
- Convolutional layer with a 3x3 filter size and a depth of 32
- Convolutional layer with a 3x3 filter size and a depth of 32

Each convolution layer is followed by a max pooling with a pool size of 2x2 and a RELU activation function.

The model then continues:

- Fully connected layer with 556 outputs
- Fully connected layer with 24 outputs

Each fully connected layer uses dropout and a RELU activation function.

Finally:

- Fully connected layer with 1 output (model output)

Here is a visualization of the architecture:

![alt text][model1]

#### 3. Creation of the Training Set & Training Process

Since I started this project about 10 days too late, I decided to crowd source my data gathering. I messaged 12 friends and sent them links to the simulator and explained to them how to record the data. I received recorded runs from 4 of them. These combined with my own runs and the Udacity sample data, I had plenty of training data to work with.

To help prevent the model from only learning to steer to one side, I recorded several laps going around the track in the opposite direction.

I augmented the data set by using the left and right cameras with a constant offset for the steering angle (parameterized via my [project3.py] wrapper script).

![alt text][left_image]
![alt text][center_image]
![alt text][right_image]

Additionally, I also flipped images and angles for any input samples that had a non-zero angle. An example is:

![alt text][normal_image]
![alt text][flipped_image]

I noticed that input samples with a steering angle of 0.0 dominated the total training data.

![alt text][histogram_all]

Because I didn't want my model to be focused on when not to turn, I implemented a parameter to ignore a percentage of input samples with a 0.0 steering angle. In my experiments, I found 80% to be close to optimal. 

![alt text][histogram_ignored80]

In the end, the model was trained on 102,816 samples and validated using 25,704 samples.

I played a lot with hyper parameters for my models, hence the creation of the [project3.py] wrapper script and the [train_the_ocean.sh] shell script. In the end, I ran over 100 experiments to arrive at my final model.
