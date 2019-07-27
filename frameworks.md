### Caffe

  Both Deeplearning4j and Caffe perform image classification with convolutional nets, which represent the state of the art.
  Caffe is not intended for other deep-learning applications such as text, sound or time series data.
  
  Pros and Cons:  https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch

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
    
  My practice
  
    (+) Uses little memory for training
    (+) The resulting models are qualitative and are used as pretrained models in other frameworks
    
    https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch

### Tensorflow
