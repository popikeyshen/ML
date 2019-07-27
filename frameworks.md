### Caffe

  Both Deeplearning4j and Caffe perform image classification with convolutional nets, which represent the state of the art.
  Caffe is not intended for other deep-learning applications such as text, sound or time series data.
  
  Pros and Cons:

    (+) Good for feedforward networks and image processing
    (+) Good for finetuning existing networks
    (+) Train models without writing any code
    (+) Python interface is pretty useful
    (-) Need to write C++ / CUDA for new GPU layers
    (-) Not good for recurrent networks
    (-) Cumbersome for big networks (GoogLeNet, ResNet)
    (-) Not extensible, bit of a hairball
    (-) No commercial support
    (-) Probably dying; slow development
