# my-Punctuator

The code contained in this repository is for adding punctuations to text which does not have any punctuations such as machine transcribed text.
This project is inspired by the work by [Ottokar Tilk and Tanel Alumae](http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1517.PDF) and [Their github repo](https://github.com/ottokart/punctuator2), and also inspired by [Andrew Ng's Deep Learning course on Coursera](https://www.coursera.org/learn/nlp-sequence-models/home/info).
This system is basically the same as the system described in [the paper](http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1517.PDF), except that this is implemented with Keras, uses different labels, does not use the audio feature, and is designed for English text.

The model, pretrained weights, and a glove word vec file can be downloaded from here:
[model](https://s3-us-west-2.amazonaws.com/models-text-and-other-data/my-punctuator/model9.json)
[weights](https://s3-us-west-2.amazonaws.com/models-text-and-other-data/my-punctuator/model9.h5)
[glove word vec file](https://s3-us-west-2.amazonaws.com/models-text-and-other-data/my-punctuator/glove.6B.50d.txt)

Usage:
1. to train
   run trainer.py

2. to test
   run punctuate.py

3. to add punctuations to text
   Use Punctuator class defined in punctuator.py.
   Refer to punctuate.py for an exapmle usage.
