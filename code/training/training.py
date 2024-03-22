from audiocraft.models import MusicGen
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf
from training_utils import *
import json
import os
import sys
import shutil
import copy
from peft import get_peft_model, LoraConfig


def get_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


def save_model(step: int, is_final_save: bool):
    logging.info(f"Model save at step {int(step)}")
    torch.save(video_to_t5.state_dict(),
               f"{model_save_path}/lm_{'final' if is_final_save else int(step)}.pt")

    if conf.use_peft:
        musicgen_model.lm.save_pretrained(f"{model_save_path}/musicgen_peft_{'final' if is_final_save else int(step)}")


conf = OmegaConf.load('training_conf.yml')
start_timestamp = get_current_timestamp()
model_save_path = f"./{'models_peft' if conf.use_peft else 'models_audiocraft'}/{conf.output_dir_name}"

if os.path.isdir(model_save_path):
    i = input(f"Model output directory {model_save_path} already exists, overwrite directory? confirm with [y]\n")
    if i == "y" or i == "yes":
        shutil.rmtree(model_save_path)
    else:
        print("Aborting.")
        sys.exit()

configure_logging(model_save_path, f"{start_timestamp}.log", conf.log_level)
os.makedirs(model_save_path, exist_ok=True)

if conf.use_wandb:
    wandb = wandb.init(project=conf.wandb_project_name,
                       config=OmegaConf.to_container(conf))
    logging.info(f"Wandb project_name: {conf.wandb_project_name}, run_id: {wandb.id}, run_name: {wandb.id}")

logging.info("Start Training")
musicgen_model = MusicGen.get_pretrained(conf.musicgen_model_id, device=conf.device)
musicgen_model.compression_model = musicgen_model.compression_model.to(conf.device)
musicgen_model.lm = musicgen_model.lm.to(conf.device)
musicgen_model.lm = musicgen_model.lm.train()

encoder_output_dimension = None
if "small" in conf.musicgen_model_id:
    encoder_output_dimension = 1024
elif "medium" in conf.musicgen_model_id:
    encoder_output_dimension = 1536
elif "large" in conf.musicgen_model_id:
    encoder_output_dimension = 2048
assert encoder_output_dimension, f"Video Encoder output dimension could not be determined by {conf.musicgen_model_id}"

# initialize video-to-text model
video_to_t5 = VideoToT5(video_extraction_framerate=conf.video_extraction_framerate,
                        encoder_input_dimension=conf.encoder_input_dimension,
                        encoder_output_dimension=encoder_output_dimension,
                        encoder_heads=conf.encoder_heads,
                        encoder_dim_feedforward=conf.encoder_dim_feedforward,
                        encoder_layers=conf.encoder_layers,
                        device=conf.device)

# freeze all model layers that except the video-to-text encoder
freeze_model(video_to_t5.video_feature_extractor)
freeze_model(musicgen_model.compression_model)
if not conf.use_peft:
    freeze_model(musicgen_model.lm)

logging.info(f"Trainable parameters video_to_t5: {get_trainable_parameters(video_to_t5)}")

if conf.use_peft:
    lora_config = LoraConfig(
        r=conf.lora_r,
        lora_alpha=conf.lora_alpha,
        target_modules=["out_proj", "linear1", "linear2"],
        lora_dropout=conf.lora_dropout,
        bias="none",
        modules_to_save=["classifier"]
    )

    logging.info(f"Trainable parameters MusicGen before LoRA: {get_trainable_parameters(musicgen_model.lm)}")
    musicgen_model.lm = get_peft_model(musicgen_model.lm, lora_config)
    logging.info(f"Trainable parameters MusicGen with LoRA: {get_trainable_parameters(musicgen_model.lm)}")

logging.info(f"Training on {conf.musicgen_model_id}")

# create dataset train and validation split
dataset = VideoDataset(conf.dataset_video_folder)
train_indices, validation_indices, test_indices = split_dataset_randomly(dataset,
                                                                         conf.dataset_validation_split,
                                                                         conf.dataset_test_split,
                                                                         seed=conf.dataset_shuffling_seed)

train_dataset = copy.copy(dataset)
train_dataset.data_map = [dataset.data_map[i] for i in train_indices]
validation_dataset = copy.copy(dataset)
validation_dataset.data_map = [dataset.data_map[i] for i in validation_indices]
test_dataset = copy.copy(dataset)
test_dataset.data_map = [dataset.data_map[i] for i in test_indices]
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=conf.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=1)

with open(f"{model_save_path}/dataset_split.json", 'w') as f:
    json.dump({
        "dataset": dataset.data_map,
        "training": train_dataloader.dataset.data_map,
        "validation": validation_dataloader.dataset.data_map,
    }, f)

with open(f"{model_save_path}/configuration.yml", 'w') as f:
    OmegaConf.save(conf, f)

logging.info(f"Video path: {conf.dataset_video_folder}, "
             f"Audio path: {conf.dataset_audio_folder} with {len(dataset)} examples, "
             f"Batch Size: {conf.batch_size}.")

optimizer = AdamW(
    video_to_t5.video_encoder.parameters(),
    betas=(conf.beta1, conf.beta2),
    weight_decay=conf.weight_decay,
    lr=conf.learning_rate
)


def forward_pass(video_path_list: [str]):
    optimizer.zero_grad()

    # get corresponding audio for the video data
    audio_paths = []
    for video_path in video_paths:
        # load corresponding audio file
        _, video_file_name = os.path.split(video_path)
        video_file_name = video_file_name[:-4]  # remove .mp4
        if conf.use_demucs_folder_structure:
            audio_path = f"{conf.dataset_audio_folder}/htdemucs/{video_file_name}/no_vocals.wav"
        else:
            audio_path = f"{conf.dataset_audio_folder}/{video_file_name}.wav"
        audio_paths.append(audio_path)

    # batch encode audio data
    audio_batches = generate_audio_codes(audio_paths=audio_paths,
                                         audiocraft_compression_model=musicgen_model.compression_model,
                                         device=conf.device)

    # batch encode video data
    video_embedding_batches = video_to_t5(video_path_list)

    condition_tensors = create_condition_tensors(video_embedding_batches,
                                                 conf.batch_size,
                                                 video_to_t5.video_extraction_framerate,
                                                 device=conf.device)

    # forward pass with MusicGen
    with musicgen_model.autocast:
        musicgen_output = musicgen_model.lm.compute_predictions(
            codes=audio_batches,
            conditions=[],
            condition_tensors=condition_tensors
        )
    loss, _ = compute_cross_entropy(logits=musicgen_output.logits,
                                    targets=audio_batches,
                                    mask=musicgen_output.mask)
    return musicgen_output, loss


training_step = 0
for epoch in range(conf.num_epochs):
    epoch_training_loss = []
    epoch_validation_loss = []
    logging.info("Starting next Epoch.")
    for batch_idx, video_paths in enumerate(train_dataloader):
        _, training_loss = forward_pass(video_paths)
        epoch_training_loss.append(training_loss)

        training_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(musicgen_model.lm.parameters(), conf.gradient_clipping)
        optimizer.step()
        training_step += 1

        # update metrics
        if conf.use_wandb:
            wandb.log(dict(training_loss=training_loss.item()))
        logging.info(
            f"Epoch: {epoch + 1}/{conf.num_epochs}, "
            f"Batch: {batch_idx}/{len(train_dataloader)}, "
            f"Loss: {training_loss.item()}"
        )

    # save model after each epoch
    save_model(training_step, False)

    # testing
    logging.info("Start Validation.")
    with torch.no_grad():
        for batch_idx, video_paths in enumerate(validation_dataloader):
            _, validation_loss = forward_pass(video_paths)
            epoch_validation_loss.append(validation_loss)
            if conf.use_wandb:
                wandb.log(dict(validation_loss=validation_loss.item()))
            logging.info(
                f"Epoch: {epoch + 1}/{conf.num_epochs}, "
                f"Batch: {batch_idx}/{len(validation_dataloader)}, "
                f"Loss: {validation_loss.item()}"
            )
    logging.info(
        f"Epoch results: epoch_training_loss {epoch_training_loss}, epoch_validation_loss {epoch_validation_loss}")
save_model(training_step, True)
logging.info(f"Finished Training. Start Testing")
with torch.no_grad():
    for batch_idx, video_paths in enumerate(test_dataloader):
        _, testing_loss = forward_pass(video_paths)
        if conf.use_wandb:
            wandb.log(dict(testing_loss=testing_loss.item()))
        logging.info(
            f"Epoch: {epoch + 1}/{conf.num_epochs}, "
            f"Batch: {batch_idx}/{len(test_dataloader)}, "
            f"Loss: {testing_loss.item()}"
        )
logging.info(f"Finished Testing.")
