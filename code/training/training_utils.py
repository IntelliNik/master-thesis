from torch.utils.data import Dataset
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio
import os
import logging
from torchvision.models import resnet50, ResNet50_Weights, resnet152, resnet18, resnet34, ResNet152_Weights
from PIL import Image
from time import strftime
import math
import numpy as np
import moviepy.editor as mpe


class VideoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, extension = os.path.splitext(d)
            if extension == ".mp4":
                self.data_map.append({"video": os.path.join(data_dir, d)})

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        return self.data_map[idx]["video"]


# input: video_path, output: wav_music
class VideoToT5(nn.Module):
    def __init__(self,
                 device: str,
                 video_extraction_framerate: int,
                 encoder_input_dimension: int,
                 encoder_output_dimension: int,
                 encoder_heads: int,
                 encoder_dim_feedforward: int,
                 encoder_layers: int
                 ):
        super().__init__()
        self.video_extraction_framerate = video_extraction_framerate
        self.video_feature_extractor = VideoFeatureExtractor(video_extraction_framerate=video_extraction_framerate,
                                                             device=device)
        self.video_encoder = VideoEncoder(
            device,
            encoder_input_dimension,
            encoder_output_dimension,
            encoder_heads,
            encoder_dim_feedforward,
            encoder_layers
        )

    def forward(self, video_paths: [str]):
        image_embeddings = []
        for video_path in video_paths:
            video = mpe.VideoFileClip(video_path)
            video_embedding = self.video_feature_extractor(video)
            image_embeddings.append(video_embedding)
        video_embedding = torch.stack(
            image_embeddings)  # resulting shape: [batch_size, video_extraction_framerate, resnet_output_dimension]
        # not used, gives worse results!
        # video_embeddings = torch.mean(video_embeddings, 0, True)  # average out all image embedding to one video embedding

        t5_embeddings = self.video_encoder(video_embedding)  # T5 output: [batch_size, num_tokens,
        # t5_embedding_size]
        return t5_embeddings


class VideoEncoder(nn.Module):
    def __init__(self,
                 device: str,
                 encoder_input_dimension: int,
                 encoder_output_dimension: int,
                 encoder_heads: int,
                 encoder_dim_feedforward: int,
                 encoder_layers: int
                 ):
        super().__init__()
        self.device = device
        self.encoder = (nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_input_dimension,
                nhead=encoder_heads,
                dim_feedforward=encoder_dim_feedforward
            ),
            num_layers=encoder_layers,
        )
        ).to(device)

        # linear layer to match T5 embedding dimension
        self.linear = (nn.Linear(
            in_features=encoder_input_dimension,
            out_features=encoder_output_dimension)
                       .to(device))

    def forward(self, x):
        assert x.dim() == 3
        x = torch.transpose(x, 0, 1)  # encoder expects [sequence_length, batch_size, embedding_dimension]
        x = self.encoder(x)  # encoder forward pass
        x = self.linear(x)  # forward pass through the linear layer
        x = torch.transpose(x, 0, 1)  # shape: [batch_size, sequence_length, embedding_dimension]
        return x


class VideoFeatureExtractor(nn.Module):
    def __init__(self,
                 device: str,
                 video_extraction_framerate: int = 1,
                 resnet_output_dimension: int = 2048):
        super().__init__()
        self.device = device

        # using a ResNet trained on ImageNet
        self.resnet = resnet50(weights="IMAGENET1K_V2").eval()
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])).to(device)  # remove ResNet layer
        self.resnet_preprocessor = ResNet50_Weights.DEFAULT.transforms().to(device)
        self.video_extraction_framerate = video_extraction_framerate  # setting the fps at which the video is processed
        self.positional_encoder = PositionalEncoding(resnet_output_dimension).to(device)

    def forward(self, video: mpe.VideoFileClip):
        embeddings = []
        for i in range(0, 30 * self.video_extraction_framerate):
            i = video.get_frame(i)  # get frame as numpy array
            i = Image.fromarray(i)  # create PIL image from numpy array
            i = self.resnet_preprocessor(i)  # preprocess image
            i = i.to(self.device)
            i = i.unsqueeze(0)  # adding a batch dimension
            i = self.resnet(i).squeeze()  # ResNet forward pass
            i = i.squeeze()
            embeddings.append(i)  # collect embeddings

        embeddings = torch.stack(embeddings)  # concatenate all frame embeddings into one video embedding
        embeddings = embeddings.unsqueeze(1)
        embeddings = self.positional_encoder(embeddings)  # apply positional encoding with a sequence length of 30
        embeddings = embeddings.squeeze()
        return embeddings


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(30).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(30, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def split_dataset_randomly(dataset, validation_split: float, test_split: float, seed: int = None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    datapoints_validation = int(np.floor(validation_split * dataset_size))
    datapoints_testing = int(np.floor(test_split * dataset_size))

    if seed:
        np.random.seed(seed)

    np.random.shuffle(indices)  # in-place operation
    training = indices[datapoints_validation + datapoints_testing:]
    validation = indices[datapoints_validation:datapoints_testing + datapoints_validation]
    testing = indices[:datapoints_testing]

    assert len(validation) == datapoints_validation, "Validation set length incorrect"
    assert len(testing) == datapoints_testing, "Testing set length incorrect"
    assert len(training) == dataset_size - (datapoints_testing + datapoints_testing), "Training set length incorrect"
    assert not any([item in training for item in validation]), "Training and Validation overlap"
    assert not any([item in training for item in testing]), "Training and Testing overlap"
    assert not any([item in validation for item in testing]), "Validation and Testing overlap"

    return training, validation, testing


### private function from audiocraft.solver.musicgen.py => _compute_cross_entropy
def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """Compute cross entropy between multi-codebook targets and model's logits.
    The cross entropy is computed per codebook to provide codebook-level cross entropy.
    Valid timesteps for each of the codebook are pulled from the mask, where invalid
    timesteps are set to 0.

    Args:
        logits (torch.Tensor): Model's logits of shape [B, K, T, card].
        targets (torch.Tensor): Target codes, of shape [B, K, T].
        mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
    Returns:
        ce (torch.Tensor): Cross entropy averaged over the codebooks
        ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
    """
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook = []
    for k in range(K):
        logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        ce += q_ce
        ce_per_codebook.append(q_ce.detach())
    # average cross entropy across codebooks
    ce = ce / K
    return ce, ce_per_codebook


def generate_audio_codes(audio_paths: [str],
                         audiocraft_compression_model: torch.nn.Module,
                         device: str) -> torch.Tensor:
    audio_duration = 30
    encodec_sample_rate = audiocraft_compression_model.sample_rate

    torch_audios = []
    for audio_path in audio_paths:
        wav, original_sample_rate = torchaudio.load(audio_path)  # load audio from file
        wav = torchaudio.functional.resample(wav, original_sample_rate,
                                             encodec_sample_rate)  # cast audio to model sample rate
        wav = wav[:, :encodec_sample_rate * audio_duration]  # enforce an exact audio length of 30 seconds

        assert len(wav.shape) == 2, f"audio data is not of shape [channels, duration]"
        assert wav.shape[0] == 2, "audio data should be in stereo, but has not 2 channels"

        torch_audios.append(wav)

    torch_audios = torch.stack(torch_audios)
    torch_audios = torch_audios.to(device)

    with torch.no_grad():
        gen_audio = audiocraft_compression_model.encode(torch_audios)

    codes, scale = gen_audio
    assert scale is None

    return codes


def create_condition_tensors(
        video_embeddings: torch.Tensor,
        batch_size: int,
        video_extraction_framerate: int,
        device: str
):
    mask = torch.ones((batch_size, video_extraction_framerate * 30), dtype=torch.int).to(device)

    condition_tensors = {
        'description': (video_embeddings, mask)
    }
    return condition_tensors


def get_current_timestamp():
    return strftime("%Y_%m_%d___%H_%M_%S")


def configure_logging(output_dir: str, filename: str, log_level):
    # create logs folder, if not existing
    os.makedirs(output_dir, exist_ok=True)
    level = getattr(logging, log_level)
    file_path = output_dir + "/" + filename
    logging.basicConfig(filename=file_path, encoding='utf-8', level=level)
    logger = logging.getLogger()
    # only add a StreamHandler if it is not present yet
    if len(logger.handlers) <= 1:
        logger.addHandler(logging.StreamHandler())
