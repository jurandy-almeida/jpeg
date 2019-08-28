# Getting Started

This document briefly describes how to install and use the code.


## Environment

We tested this code in the following environment:
 - Linux
 - Python 3
 - Libjpeg

Similar environments (e.g. with OSX, Python 2) might work with small modification, but not tested.


## Description

This is a python wrapper that opens a JPEG image and extracts the DCT coefficients as a numpy array.


#### Install

 - Download Libjpeg (`curl https://www.ijg.org/files/jpegsrc.v9c.tar.gz | tar xvz`).
 - Go to Libjpeg home
 - `make clean`
 - `./configure --prefix=${LIBJPEG_INSTALL_PATH} --enable-shared`
 - `make`
 - `make install`
 - If needed, add `${LIBJPEG_INSTALL_PATH}/lib/` to `$LD_LIBRARY_PATH`.
 - Modify `setup.py` to use your Libjpeg path (`${LIBJPEG_INSTALL_PATH}`).
 - `./install.sh`


#### Usage

The python wrapper has two functions: `load` and `save`.

The following call loads JPEG raw data into a numpy array.
```python
from jpeg import load
load([input], [decode])
```
 - input: path to a JPEG image.
 - decode: `True` or `False`. `True` decodes the JPEG raw data and returns the RGB pixels. `False` parses the JPEG raw data and returns the DCT coefficients.

For example, 
```
load('input.jpg', False)
```

The following call saves a numpy array as a JPEG image.
```python
from jpeg import save
save([array], [output])
```
 - output: path to a JPEG image.
 - array: a numpy array with shape `height` x `width` x 3

For example, 
```
imarr = numpy.random.randint(255, size=(100,100,3), dtype='uint8')
save('output.jpg', imarr)
```
