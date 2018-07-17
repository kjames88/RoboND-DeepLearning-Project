## Follow Me Deep Learning Project ##

### Goal

The goal of this project is to create and train a Fully Convolutional Network for semantic segmentation.  The camera images from a virtual drone will be segmented into background, other people, and hero.  The hero is the target for the drone to follow.

### Network Architecture

Per the lesson lab, the network is composed series of separable 2D convolution layers utilizing batch normalization at each layer.  These _encoder_ layers use kernel size 3, _relu_ activation, and a stride of 2 with _same_ padding to yield an output dimension ratio of 0.5 on each axis.  Separable convolution layers are used to reduce the number of parameters in the network.  Following the convolutions, a 1x1 convolution layer is added.  Whereas a typical classifier would use fully connected layer(s) here to generate a translation invariant estimate of the class present in the scene, the 1x1 convolution preserves spatial information at the resolution of the final convolution layer.  The following layers each upsample by 2x on each axis using bilinear upsampling.  The number of these layers matches the number of convolution layers, so the final resolution matches that of the input.  Each of these layers also takes as input the output of its matched resolution encoder layer and concatenates this with the upsampled input.  This skip input brings higher resolution information.  Finally, each decoder layer uses a separable convolution with stride 1 to produce its result.  The final network output is a 2D convolution with _softmax_ activation.  This produces a pixel-level classification of the image.


Following is the network architecture:


| Layer | Type | Kernel | Stride | Filters |
| ----- | ---- | ------ | ------ | ------- |
| 1     | separable conv 2D relu | 3x3 | 2 | 32 |
| 2     | separable conv 2D relu | 3x3 | 2 | 128 |
| 2d    | dropout 0.5 | na | na | na |
| 3     | separable conv 2D relu | 3x3 | 2 | 512 |
| 3d    | dropout 0.5 | na | na | na |
| 4     | conv 2D relu | 1x1 | 1 | 1024 |
| 5     | decoder | 3x3 | 1 | 256 |
| 6     | decoder | 3x3 | 1 | 64 |
| 7     | decoder | 3x3 | 1 | 16 |
| out   | conv 2D softmax | 3x3 | 1 | 3  |


Following is the structure of the decoder layer:


| Decoder Layer | Type |
| ------------- | ---- |
| 1 | bilinear upsample 2x |
| 2 | concat |
| 3 | conv 2D relu 3x3 stride 1 |

