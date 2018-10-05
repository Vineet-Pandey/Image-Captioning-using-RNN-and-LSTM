# Image-Captioning-using-RNN-and-LSTM
In this project, I have created a neural network architecture to automatically generate captions from images.
Microsoft Common Objects in COntext (MS COCO) dataset has been used to train the network.

LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!
![alt text](https://github.com/Vineet-Pandey/Image-Captioning-using-RNN-and-LSTM/blob/master/Image%20Captioning/image-captioning.png)

Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are embed_size.

Note: Create a folder named 'Models' before executing the code and training the network. Both Encoder and Decoder will generate a PKL file each corresponding to the number of epochs chosen for training. Epochs=3 gives good results. However, significant improvements will be seen if trained more. Kindly note training this network will take longer hours- EVEN ON GPU!!!
