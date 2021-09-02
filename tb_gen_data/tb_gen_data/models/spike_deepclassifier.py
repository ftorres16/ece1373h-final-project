from enum import Enum

import torch
from torch import nn

from tb_gen_data.models.bar import BAR
from tb_gen_data.models.spike_deeptector import SpikeDeeptector
from tb_gen_data.models.pca import PCAModel

THRESHOLD = 0.85


class ChannelType(Enum):
    NEURAL = 0
    NOISE = 1


class SampleType(Enum):
    SPIKE = 0
    BACKGROUND = 1


class SpikeDeepClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.spike_deeptector = SpikeDeeptector()
        self.bar = BAR()
        self.pca = PCAModel()

    def forward(self, input_):
        self.spike_deeptector_outputs = [
            self.spike_deeptector(electrode) for electrode in input_
        ]

        self.channel_labels = []
        for channel in self.spike_deeptector_outputs:
            chunk_labels = torch.argmax(channel, dim=1)

            label = (
                ChannelType.NEURAL
                if torch.sum(chunk_labels == ChannelType.NEURAL.value)
                > THRESHOLD * len(chunk_labels)
                else ChannelType.NOISE
            )
            self.channel_labels.append(label)

        self.bar_outputs = [
            self.bar(sample)
            for electrode, label in zip(input_, self.channel_labels)
            for chunk in torch.split(electrode, 1)
            for sample in torch.split(chunk, 1, dim=2)
            if label == ChannelType.NEURAL
        ]

        self.bar_labels = [torch.argmax(logits).item() for logits in self.bar_outputs]
        self.bar_labels = [
            SampleType.SPIKE
            if label == SampleType.SPIKE.value
            else SampleType.BACKGROUND
            for label in self.bar_labels
        ]

        self.pca_outputs = [
            torch.reshape(self.pca(torch.reshape(electrode, (-1, 48))), (-1, 1, 20, 48))
            for electrode, label in zip(input_, self.channel_labels)
            if label == ChannelType.NEURAL
        ]

        return self.pca_outputs
