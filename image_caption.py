import os
import copy
import math
from collections import defaultdict
import numpy as np
import PIL
from keras import Sequential, Model
from keras.layers import Embedding, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from keras.activations import softmax
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam

# if GPU is present, use CudnnLSTM, else
from keras.layers import LSTM
# from keras.layers import CuDNNLSTM as LSTM

import matplotlib
matplotlib.use('qt5agg')  # environment-dependent
from matplotlib import pyplot as plt

DATA_PATH = "./data/Flickr8k_text/"
IMG_PATH = "./data/Flicker8k_Dataset/"
ENCODING_PATH = "./img_encodings/"
MODEL_PATH = "./models/"
MODEL_FILES = ["model.h5", "model_CPU.h5"]

class ImageCaptioner:

    def __init__(self):

        self.load_dataset()
        try:
            self.load_encoded_images()
        except FileNotFoundError:
            self.encode_images()

        self.img_to_enc = self.build_img_indices([self.train_list, self.dev_list, self.test_list],
                                                 [self.enc_train, self.enc_dev, self.enc_test])

        self.descriptions = self.read_image_descriptions()
        self.id_to_word, self.word_to_id = self.build_indices()

        self.MAX_LEN = max(len(description) for img_id in self.train_list for description in self.descriptions[img_id])
        self.vocab_size = len(self.word_to_id)

    def load_image_list(self, filename):
        with open(filename, 'r') as img_list_f:
            return [line.strip() for line in img_list_f]

    def load_dataset(self):
        self.train_list = self.load_image_list(DATA_PATH + 'Flickr_8k.trainImages.txt')
        self.dev_list = self.load_image_list(DATA_PATH + 'Flickr_8k.devImages.txt')
        self.test_list = self.load_image_list(DATA_PATH + 'Flickr_8k.testImages.txt')

    def get_image(self, img_name, original=False, dataset=True):
        if dataset:
            path = os.path.join(IMG_PATH, img_name)
        else:
            path = img_name

        img = PIL.Image.open(path)
        if original:
            return img
        else:
            return np.asarray(img.resize((299, 299))) / 255.0

    def img_generator(self, img_list, dataset=True):
        for img in img_list:
            yield np.array([self.get_image(img, dataset=dataset)])

    # encode dataset into ndarrays
    def encode_images(self):
        img_model = InceptionV3(weights='imagenet')
        img_encoder = Model(img_model.input, img_model.layers[-2].output)

        print("Encoding images...")
        self.enc_train = img_encoder.predict_generator(self.img_generator(self.train_list),
                                                       steps=len(self.train_list), verbose=1)
        self.enc_dev = img_encoder.predict_generator(self.img_generator(self.dev_list),
                                                     steps=len(self.dev_list), verbose=1)
        self.enc_test = img_encoder.predict_generator(self.img_generator(self.test_list),
                                                      steps=len(self.test_list), verbose=1)

        print("Saving encodings...")
        np.save(ENCODING_PATH + "encoded_images_train.npy", self.enc_train)
        np.save(ENCODING_PATH + "encoded_images_dev.npy", self.enc_dev)
        np.save(ENCODING_PATH + "encoded_images_test.npy", self.enc_test)

    # encode individual images
    def encode_image(self, img_path):
        try:
            f = open(img_path, 'r')
        except FileNotFoundError:
            raise Exception("Cannot encode image. Invalid file path.")

        print("Encoding image..")
        img_model = InceptionV3(weights='imagenet')
        img_encoder = Model(img_model.input, img_model.layers[-2].output)
        enc_img = img_encoder.predict_generator(self.img_generator([img_path], dataset=False), steps=1, verbose=1)

        return enc_img[0]

    def load_encoded_images(self):
        self.enc_train = np.load(ENCODING_PATH + "encoded_images_train.npy")
        self.enc_dev = np.load(ENCODING_PATH + "encoded_images_dev.npy")
        self.enc_test = np.load(ENCODING_PATH + "encoded_images_test.npy")

    def build_img_indices(self, img_lists, enc_img_lists):
        img_to_enc = defaultdict()
        for img_list, enc_imgs in zip(img_lists, enc_img_lists):
            for img, enc_img in zip(img_list, enc_imgs):
                img_to_enc[img] = enc_img

        return img_to_enc

    def show_image(self, img_name, original=True):
        try:
            img = self.get_image(img_name, original=original)
            plt.imshow(img)
            plt.show()
        except FileNotFoundError as e:
            print(e)

    # read image descriptions into dict
    def read_image_descriptions(self):
        img_descriptions = defaultdict(list)
        file = open(DATA_PATH + "Flickr8k.token.txt", 'r')
        for line in file:
            img_name = line[:line.find('#')]

            description = line[line.find('\t'):].lower().strip('\t\n').split()
            description.insert(0, '<START>')
            description.append('<END>')

            img_descriptions[img_name].append(description)

        return img_descriptions

    def build_indices(self):
        tokens = set()
        for img, descs in self.descriptions.items():
            for desc in descs:
                for word in desc:
                    tokens.add(word)

        tokens = sorted(list(tokens))

        id_to_word = dict(enumerate(tokens))
        word_to_id = {v: k for k, v in id_to_word.items()}

        return id_to_word, word_to_id

    def get_img_input_representation(self, index):
        return self.enc_train[index]

    def get_text_input_representation(self, desc):
        arr = [self.word_to_id[word] for word in desc]
        arr = np.pad(arr, (0, self.MAX_LEN - len(arr)), 'constant')
        return arr

    def get_output_representation(self, word):
        return to_categorical([self.word_to_id[word]], num_classes=self.vocab_size)

    def training_generator(self, img_list, batch_size=128):
        img_inputs = []
        text_inputs = []
        outputs = []
        while True:
            for index, img_name in enumerate(img_list):
                for description in self.descriptions[img_name]:
                    for i in range(1, len(description)):
                        img_inputs.append(self.get_img_input_representation(index))
                        text_inputs.append(self.get_text_input_representation(description[:i]))
                        outputs.append(self.get_output_representation(description[i]))

                        if len(outputs) == batch_size:
                            yield [np.vstack(img_inputs), np.vstack(text_inputs)], np.vstack(outputs)
                            img_inputs = []
                            text_inputs = []
                            outputs = []

    def build_model(self):
        print("Building model...")
        self.IMG_DIM = 2048
        EMBEDDING_DIM = 300
        IMAGE_ENC_DIM = 300

        # Image input
        img_input = Input(shape=(self.IMG_DIM,))
        img_enc = Dense(IMAGE_ENC_DIM, activation="relu")(img_input)
        images = RepeatVector(self.MAX_LEN)(img_enc)

        # Text input
        text_input = Input(shape=(self.MAX_LEN,))
        embedding = Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.MAX_LEN)(text_input)
        x = Concatenate()([images, embedding])
        y = Bidirectional(LSTM(256, return_sequences=False))(x)
        pred = Dense(self.vocab_size, activation='softmax')(y)
        self.model = Model(inputs=[img_input, text_input], outputs=pred)
        self.model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])

        return self.model

    def fit_and_save_model(self, filename='model.h5', batch_size=128):
        if not hasattr(self, 'model'):
            self.build_model()

        print("Fitting model...")
        generator = self.training_generator(self.train_list, batch_size)
        steps = len(self.train_list) * self.MAX_LEN // batch_size

        self.model.fit_generator(generator, steps_per_epoch=steps, verbose=True, epochs=20)
        self.model.save_weights(MODEL_PATH + filename)

        if filename not in MODEL_FILES:
            MODEL_FILES.append(filename)

    def load_model(self, model_no=0):
        self.model = self.build_model()

        print("Loading weights...")
        if model_no not in range(len(MODEL_FILES) + 1):
            print("Invalid model number... Using default")
            model_no = 0
        self.model.load_weights(MODEL_PATH + MODEL_FILES[model_no])


    def greedy_img_decoder(self, img):
        if img in self.train_list + self.dev_list + self.test_list:
            enc_img = self.img_to_enc[img]
        else:
            try:
                enc_img = self.encode_image(img)
            except Exception as e:
                print(e)
                return
        enc_img = enc_img.reshape(-1, self.IMG_DIM)

        inputs = ['<START>']
        while True:
            output = self.model.predict([enc_img,
                                         self.get_text_input_representation(inputs).reshape(-1, self.MAX_LEN)])
            output = self.id_to_word[np.argmax(output)]

            if output == "<END>" or len(inputs) == self.MAX_LEN:
                inputs.append("<END>")
                break

        return inputs


    def img_beam_decoder(self, img, n=5):
        if self.img_to_enc.get(img):
            enc_img = self.img_to_enc[img]
        else:
            try:
                enc_img = self.encode_image(img)
                self.img_to_enc[img] = enc_img
            except Exception as e:
                print(e)
                return
        enc_img = enc_img.reshape(-1, self.IMG_DIM)

        # initialization
        possibilities = []
        start = ['<START>']
        output = self.model.predict([enc_img,
                                     self.get_text_input_representation(start).reshape(-1, self.MAX_LEN)])
        output = np.float64(output[0])
        output = output / np.sum(output)  # normalize values
        sample = np.random.multinomial(self.vocab_size, output)
        ind_best = sample.argsort()[-n:][::-1]  # n best candidates
        for ind in ind_best:
            possibilities.append((0.0, ['<START>', self.id_to_word[ind]]))

        while max(len(seq) for p, seq in possibilities) < self.MAX_LEN:
            novel_sequences = 0  # to prevent infinite loop
            new_poss = []

            for poss in possibilities:
                prob = poss[0]
                seq = poss[1]
                if seq[-1] == '<END>':
                    new_poss.append(poss)
                    continue

                output = self.model.predict([enc_img,
                                             self.get_text_input_representation(seq).reshape(-1, self.MAX_LEN)])
                output = np.float64(output[0])
                output = output / np.sum(output)  # normalize values

                sample = np.random.multinomial(self.vocab_size, output)
                sample = sample / np.sum(sample)  # convert to probabilities

                ind_best = sample.argsort()[-n:][::-1]  # n best candidates
                for ind in ind_best:
                    if sample[ind] == 0.0:
                        continue
                    new_seq = copy.deepcopy(seq)
                    new_seq.append(self.id_to_word[ind])
                    new_prob = math.fsum([prob, math.log10(sample[ind])])

                    new_poss.append((new_prob, new_seq))
                    novel_sequences += 1

            new_poss = sorted(new_poss, key=lambda x: x[0])  # sort by log probabilities
            possibilities = new_poss[-n:][::-1]  # select n most probable sequences

            if novel_sequences == 0:
                break

        sequences = [s for p, s in possibilities]

        sentences = []
        for seq in sequences:
            seq = seq[1:-1]
            temp = list(seq[0])
            temp[0] = temp[0].upper()
            seq[0] = ''.join(temp)
            sentences.append(' '.join(seq))

        return sentences


if __name__ == "__main__":
    captioner = ImageCaptioner()
    captioner.load_model()

    # model_file = 'model_new.h5'
    # captioner.fit_and_save_model(filename=model_file)
