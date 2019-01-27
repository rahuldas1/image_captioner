# Image Captioning

This program uses the [Flickr8k dataset](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html) to train an LSTM language generator on the caption data. For image encoding, the program uses the pre-trained CNN InceptionV3.\
For decoding, I have used two approaches -- greedy_decoder() simply selects the most probable next token to construct the caption sequence; img_beam_decoder() uses beam search to explore the most favorable candidates, pruning the list at each step and returning the best *n* candidates once a certain max length is encountered, or when there are no new sequences being generated.

# Models

model.h5 was trained using CuDNNLSTM.\
In case of compatibility issues, model weights can be loaded from model_CPU.h5, which was trained using standard LSTM.
I will continue to update the model files as I retrain and improve the models.

# Necessary packages

numpy\
keras\
PIL (pip install pillow)\
matplotlib

# Some examples

| ![dogs](/test_pics/3385593926_d3e9c21170.jpg) |
|:--:|
| *Two dogs are playing in the snow* |

| ![dog](/test_pics/2677656448_6b7e7702af.jpg) |
|:--:|
| *The dog swims in the water* |

| ![more dogs](/test_pics/2723477522_d89f5ac62b.jpg) |
|:--:|
| *Two dogs run through the grass together* |

| ![children](/test_pics/2844018783_524b08e5aa.jpg) |
|:--:|
| *A group of kids have their arms around each other as they sit on the ground* |

| ![crows](/test_pics/3100251515_c68027cc22.jpg) |
|:--:|
| *A man and a woman in a crowd* |

| ![climber](/test_pics/872622575_ba1d3632cc.jpg) |
|:--:|
| *This climber is climbing a steep rock* |

| ![ball](/test_pics/3222055946_45f7293bb2.jpg) |
|:--:|
| *A man in a white shirt is playing a ball* |

| ![crowd](/test_pics/1174629344_a2e1a2bdbf.jpg) |
|:--:|
| *A man in a blue shirt and a woman in a black shirt in a crowd* |

| ![swing](/test_pics/3453259666_9ecaa8bb4b.jpg) |
|:--:|
| *Young boy in blue shirt on a swing* |

| ![moto](/test_pics/3601843201_4809e66909.jpg) |
|:--:|
| *A motorcycle rider rides down a racetrack during a turn* |
