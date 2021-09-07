
# Pooling layers or block reduce functions

Scikit-image, keras, pytorch have block reduce.
Keras has maxpooling, averagepooling techniques.
With pytorch is possible to use power-average pooling, fractional or adaptive pooling layers.

With scikit image and numpy statistics is possible to use median pooling, minimum, percentile, range, mean, std, etc. 
```
https://numpy.org/doc/stable/reference/routines.statistics.html
https://keras.io/api/layers/
```
<p align="center"> <img src="cat.jpg"  width = 40%  /> </p>

		avgpooling, maxpooling, medianpooling 10x block, 40x and fool sized
<p align="center"> <img src="avgpool10x.png"  width = 20%  /><img src="maxpool10x.png"  width = 20%  /><img src="medianpool10x.png"             width = 20%  /> </p>
<p align="center"> <img src="avgpool40x.png"  width = 20%  /><img src="maxpool40x.png"  width = 20%  /><img src="medianpool40x.png"             width = 20%  /> </p>
<p align="center"> <img src="avgpoolfullsize.png"  width = 20%  /><img src="maxpoolfullsize.png"  width = 20%  /><img src="medianpoolfullsize.png"             width = 20%  /> </p>

