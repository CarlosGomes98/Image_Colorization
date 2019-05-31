# Image Colorization

Automatic image colourization is the task of introducing colour to a black and white
image in a realistic fashion, without any aid from the user. This has long been the
focus of research, partly due to its possible applications in image processing and
animation.

Recently, deep learning methods, specifically convolutional neural networks, have
shown great promise in image processing tasks, as image datasets become more
representative and computational power becomes more ubiquitously available. In
this work, I focus on the application of these networks to the problem of image
colourization, implementing three very different deep neural networks.

I begin with a swift introduction to neural networks and an overview of the problem
at hand. I then present three models, discussing my implementations of them
as well as their inner workings. Firstly, I develop a baseline convolutional neural
network, aiming to display the particular difficulties posed by image colourization.
Furthermore, I showcase a previously proposed state of the art classication model
able to reliably produce realistic colourings. Finally, I adapt a Deep Convolutional
Generative Adversarial Network to the task, a technique which is yet to see much
experimentation for automatic image colorization.

I gain substantial insight from the first model, confirm the excellent performance
of the second method on the Oxford102 and Places365 dataset and deliver promising
results with the GAN approach, establishing a possible way forward on this task.

### Contents

This repository contains several implementations of methods to preform colorization:

  #### 1. A deep CNN with a MSE loss  
  #### 2. A deep CNN using categorical cross entropy loss (Based on http://richzhang.github.io/colorization/)
  #### 3. A GAN
  
### Results

![demo](https://raw.githubusercontent.com/CarlosGomes98/Image_Colorization/master/diagrams/classvsmse.png)

![demo](https://raw.githubusercontent.com/CarlosGomes98/Image_Colorization/master/diagrams/allvsall.png)

