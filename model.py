import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        # initalizing the inherited nn.Module Class
        super().__init__()
        # embedding layer
        self.word_embedding_layer = nn.Embedding(vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=0,
                            batch_first=True)

        self.linear_fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]

        captions = self.word_embedding_layer(captions)

        # Combine caption and features
        inputs = torch.cat((features.unsqueeze(1), captions), 1)
        outputs, _ = self.lstm(inputs)

        # LSTM output -> word prediction
        outputs = self.linear_fc(outputs)

        return outputs

    def sample(self, inputs, states=None, max_length=20):

        caption = list()
        caption_length = 0

        while (caption_length != max_length + 1):

            # LSTM Layer
            output, states = self.lstm(inputs, states)

            # Linear Layer
            output = self.linear_fc(output.squeeze(dim=1))
            # appending output with max prob.
            _, index = torch.max(output, 1)

            caption.append(index.cpu().numpy()[0].item())

            if index == 1:
                break

            # embedding layer
            inputs = self.word_embedding_layer(index)
            inputs = inputs.unsqueeze(1)

            caption_length += 1

        return caption