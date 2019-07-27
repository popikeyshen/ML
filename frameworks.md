### Caffe

  Both Deeplearning4j and Caffe perform image classification with convolutional nets, which represent the state of the art.
  Caffe is not intended for other deep-learning applications such as text, sound or time series data.
  
  [ Pros and Cons:](https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch)

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
  
    (+) Good for production
    (+) Many good pretrained models
    (+) Uses little memory for training
    (+) The resulting models are qualitative and are used as pretrained models in other frameworks
    (+) Works fastest for embedded systems. Maybe can run faster but need to research and compare with lib TVM vs NCNN.
    
    

### Tensorflow

  [ Pros and Cons:](https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch)

    (+) Python + Numpy
    (+) Computational graph abstraction, like Theano
    (+) Faster compile times than Theano
    (+) TensorBoard for visualization
    (+) Data and model parallelism
    (-) Slower than other frameworks
    (-) Much “fatter” than Torch; more magic
    (-) Not many pretrained models
    (-) Computational graph is pure Python, therefore slow
    (-) No commercial support
    (-) Drops out to Python to load each new training batch
    (-) Not very toolable
    (-) Dynamic typing is error-prone on large software projects
    
   My practice
   
    (+) Can fast train an test model with little dataset but it will slow working - good for prototyping


### PyTorch

[ Pros and Cons:](https://skymind.ai/wiki/comparison-frameworks-dl4j-tensorflow-pytorch)

    (+) Lots of modular pieces that are easy to combine
    (+) Easy to write your own layer types and run on GPU
    (+) Lots of pretrained models
    (-) You usually write your own training code (Less plug and play)
    (-) No commercial support
    (-) Spotty documentation
    
     My practice
     
     (+) Caffe2 is in the pytorch - need to test and work
     

### MxNet

     (+) Deeplearning4j and professor rosenbrock use it - need to test

