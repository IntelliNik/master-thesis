# Master Thesis: High-Fidelity Video Background Music Generation using Transformers
### Abstract
Current artificial intelligence music generation models are mainly controlled with a single input modality: text. Adapting these models to accept alternative input modalities extends their field of use. Video input is one such modality, with remarkably different requirements for the generation of background music accompanying it. Even though alternative methods for generating video background music exist, none achieve the music quality and diversity of the text-based models. Hence, this thesis aims to efficiently reuse text-based models' high-fidelity music generation capabilities by adapting them for video background music generation. This is accomplished by training a model to represent video information inside a format that the text-based model can naturally process. To test the capabilities of our approach, we apply two datasets for model training with various levels of variation in the visual and audio parts. We evaluate our approach by analyzing the audio quality and diversity of the results. A case study is also performed to determine the video encoder's ability to capture the video-audio relationship successfully.

This repository contains the code model training and the user interface for inference.

# Installation
- create a Python virtual environment with `Python 3.11`
- check https://pytorch.org/get-started/previous-versions/ to install `PyTorch 2.1.0` with `CUDA` on your machine
- install the other requirements: `pip install -r requirements.txt`

# Folder Structure
- `code` contains the code for model `training` and `inference` of video background music
- `code/inference/gradio_app` contains the code for the interface to generate video background music
- `datasets` contains the code to create the datasets used for training within `dataset_preparation` and the videos used for training examples in the respective `nature` and `symmv` folders
- `evaluation` contains the code used to calculate the FAD scored and the ResNet embedding anaylsis of the applied datasets
- `videos` contains examplary videos used in our case study and for the Gradio interface
- `jamovia_anova.omv` containts our statstical calculations for our ANOVA analysis of our quantitative study results.

# Training
To train the models adjust the training parameters under `training/training_conf.yml` and start training with 
`python training/training.py`. The models finished models will be stored under `models/*`.

# Inference
- download the available pretrained models from here [https://huggingface.co/schnik/video-background-music-generation/tree/main](https://huggingface.co/schnik/video-background-music-generation/tree/main)
- place the unzipped directories for the respecive pretrained model in the `/models` folder
- start the Gradio interface by running `python gradio_app/app.py`
- alternatively a hosted version of the interface displaying cached results is available on Huggingface: [https://huggingface.co/spaces/schnik/video-background-music-generator](https://huggingface.co/spaces/schnik/video-background-music-generator)

# Contact
For any questions contact me at [niklas.schulte@rwth-aachen.de](mailto:niklas.schulte@rwth-aachen.de)