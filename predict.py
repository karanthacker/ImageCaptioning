import sys
import os
print(os.getcwd())
sys.path.append('/opt/cocoapi/PythonAPI')
from PIL import Image
import torch
import pickle
from model import EncoderCNN, DecoderRNN
from torchvision import transforms

from matplotlib import  pyplot as plt
def load_vocab():
    dict_object = pickle.load(open('vocab.pkl',"rb"))
    return dict_object.idx2word


def create_model():
    encoder_file = "encoder-1.pkl"
    decoder_file = "decoder-1.pkl"
    embed_size = 256
    hidden_size = 512
    vocab_size = 9955
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()
    encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file),map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file),map_location=torch.device('cpu')))
    encoder.to('cpu')
    decoder.to('cpu')
    return encoder,decoder

def clean_sentence(output,idx2word):
    caption = list()
    sentence = ""

    for i in output:
        if (i == 1):
            continue
        caption.append(idx2word[i])

    caption = caption[1:-1]

    sentence = ' '.join(caption)

    sentence = sentence.capitalize()
    return sentence

def transform_image(transform_test,image_file):
    path = 'images/'
    image = Image.open(path+image_file)
    image = transform_test(image)
    image = torch.unsqueeze(image, dim=0)
    return image

def prediction(encoder,decoder,transform_test,idx2word,image_file_list):
    for image_file in image_file_list:
        features = encoder(transform_image(transform_test,image_file)).unsqueeze(1)
        output = decoder.sample(features)
        print(image_file+":  "+clean_sentence(output,idx2word))
        plt.show()

if __name__ == "__main__":

    sys.path.append('/opt/cocoapi/PythonAPI')

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image_file_list = ['COCO_test2014_000000000001.jpg',
                       'COCO_test2014_000000000014.jpg',
                       'COCO_test2014_000000000016.jpg',
                       'COCO_test2014_000000000027.jpg',
                       'COCO_test2014_000000000057.jpg',
                       'COCO_test2014_000000000063.jpg',
                       'COCO_test2014_000000000069.jpg',
                       'coco.JPG',
                       'food.JPG']


    encoder,decoder = create_model()
    idx2word = load_vocab()
    prediction(encoder,decoder,transform_test,idx2word,image_file_list)
