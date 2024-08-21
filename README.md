# Time Scale Network

This repository contains a minimal implementation of the Time Scale Network (TiSc Net) in PyTorch for use on new datasets. The Time Scale Network is a computationally efficient shallow neural network for time series data, originally developed for biomedical applications including analysis of EKG (heart) and EEG (brain) signals.

Please cite [this article](https://ieeexplore.ieee.org/document/10637669) in all work utilizing the Time Scale Network.

This TiSc Net implementation can be installed using the pip command for system-wide availability:

```
python3 -m pip install git+https://github.com/centerforimplantabledevices/timescalenetwork.git
```

Now you are ready to use the Time Scale Network! See the `examples` folder for a brief introduction to the library and some simple demonstrations of training layers and networks. Briefly, to create a shallow TiSc Net containing only TiSc Input ( $\Lambda=[1,14]$ ), TiSc Hidden ( $\Lambda=[2,14]$ ), and Densely connected ( $units = 300, 100, 2$ ) layers, with three independent TiSc channels feeding into one dense network, you can use the `TiscMlpN` class:

```
from timeScaleNetwork import timescaleN as tiscNetwork

model = tiscNetwork.TiscMlpN( [[2, 16384], [2, 16384], 300, 100, 2], num_tisc_channels=3, length_input=data_flattened.shape[0])
```

Alternatively, a single input or hidden layer can be included in larger networks by using the `TSC_input` or `TSC_hidden` class:

```
from timeScaleNetwork import timescaleL as tiscLayer

testLayer_input  = tiscLayer.TSC_input(  2, 16384)
testLayer_hidden = tiscLayer.TSC_hidden( 2, 16384, input_size)
```

Expanded application of this network on EKG ([MIT-BIH Dataset](https://physionet.org/content/mitdb/1.0.0/)) and EEG ([CHB-MIT Dataset](https://physionet.org/content/chbmit/1.0.0/)) data can be found here and here, respectively.

 Keep in mind these best practices when using the Time Scale Network:

* It is best to keep channel and signal length dimensions to be a power of 2 (2, 4, 8, 16, 32, 64, etc). If you have an uneven dimension length, pad with zeros.
* TiSc Net was designed for data containing either known features at multilpe time scales - such as EKG with fast QRS complexes, P-waves and T-waves, slower heart rate patterns, and even slower respiration artifacts - or unknown features at unknown time scales - such as such as EEG or ECoG data with information spread across many activaiton frequencies.
* Time series signals with well understood features and structure, such as speech signals may benefit from more targeted architecture designs that directly capture know features.
* A future update will release implementations for low-power edge device including microcontrollers in Tensorflow Lite. 