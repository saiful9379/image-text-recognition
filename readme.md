# Image sequence Recognition

Keras-based CNN+LSTM trained with CTC-loss for OCR and research paper [link](https://arxiv.org/pdf/1507.05717.pdf)



![img](img/1_P4UW-wqOMSpi82KIcq11Pw.png)
This Figure use first CNN for exraction and use LSTM for sequence generation with special CTC loss.


# Environment
- python==3.6
- keras== 2.2.4 
- tensorflow=> 1.8 
- cairocffi
- editdistance
# synthesis Data Generation
Please Download ```Roboto-Regular.ttf``` and add font directory in line number 36.
# Experiment 
## Train 
## Test
Make sure your train model path then run ```prediction.py``` script.

#### Validation Result,

![img](img/e498.png)


### pretrain model
- RCNNMode_v2.ipython [model](https://drive.google.com/open?id=13utyxPpVqa5QCkJQjoj4r264QSoh_7Xd) link

# Contributor
- Saiful islam
