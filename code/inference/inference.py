from omegaconf import OmegaConf
from peft import PeftConfig, get_peft_model 
from audiocraft.models import MusicGen
from moviepy.editor import AudioFileClip
from training_utils import *
import re
import time

re_file_name = re.compile('([^/]+$)')


def generate_background_music(video_path: str,
                              dataset: str,
                              musicgen_size: str,
                              use_stereo: bool,
                              use_peft: bool,
                              device: str,
                              musicgen_temperature: float = 1.0,
                              musicgen_guidance_scale: float = 3.0,
                              top_k_sampling: int = 250) -> str:
    start = time.time()
    model_path = "../training/"
    model_path += "models_peft" if use_peft else "models_audiocraft"
    model_path += f"/{dataset}" + f"_{musicgen_size}"

    conf = OmegaConf.load(model_path + '/configuration.yml')
    use_sampling = True if top_k_sampling > 0 else False
    video = mpe.VideoFileClip(video_path)

    musicgen_model_id = "facebook/musicgen-" + "stereo-" if use_stereo else ""
    musicgen_model_id += musicgen_size

    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)

    encoder_output_dimension = None
    if "small" in conf.musicgen_model_id:
        encoder_output_dimension = 1024
    elif "medium" in conf.musicgen_model_id:
        encoder_output_dimension = 1536
    elif "large" in conf.musicgen_model_id:
        encoder_output_dimension = 2048
    assert encoder_output_dimension, f"Video Encoder output dimension could not be determined by {conf.musicgen_model_id}"

    musicgen_model = MusicGen.get_pretrained(musicgen_model_id)
    musicgen_model.lm.to(device)
    musicgen_model.compression_model.to(device)
    if use_peft:
        peft_path = model_path + "/musicgen_peft_final"
        peft_config = PeftConfig.from_pretrained(peft_path)
        musicgen_model.lm = get_peft_model(musicgen_model.lm, peft_config)
        musicgen_model.lm.load_adapter(peft_path, "default")

    print("MusicGen Model loaded.")

    video_to_t5 = VideoToT5(
        video_extraction_framerate=conf.video_extraction_framerate,
        encoder_input_dimension=conf.encoder_input_dimension,
        encoder_output_dimension=encoder_output_dimension,
        encoder_heads=conf.encoder_heads,
        encoder_dim_feedforward=conf.encoder_dim_feedforward,
        encoder_layers=conf.encoder_layers,
        device=device
    )

    video_to_t5.load_state_dict(torch.load(model_path + "/lm_final.pt", map_location=device))
    print("Video Encoder Model loaded.")

    print("Starting Video Feature Extraction.")
    video_embedding_t5 = video_to_t5(video_paths=[video_path])

    condition_tensors = create_condition_tensors(
        video_embeddings=video_embedding_t5,
        batch_size=1,
        video_extraction_framerate=video_to_t5.video_extraction_framerate,
        device=device
    )

    musicgen_model.generation_params = {
        'max_gen_len': int(video.duration * musicgen_model.frame_rate),
        'use_sampling': use_sampling,
        'temp': musicgen_temperature,
        'cfg_coef': musicgen_guidance_scale,
        'two_step_cfg': False,
    }
    if use_sampling:
        musicgen_model.generation_params['top_k'] = 250

    print("Starting Audio Generation.")
    prompt_tokens = None
    with torch.no_grad():
        with musicgen_model.autocast:
            gen_tokens = musicgen_model.lm.generate(prompt_tokens, [], condition_tensors, callback=None,
                                                    **musicgen_model.generation_params)
        gen_audio = musicgen_model.compression_model.decode(gen_tokens)

    end = time.time()
    print("Elapsed time for generation: " + str(end - start))

    _, video_file_name = os.path.split(video_path)
    video_file_name = video_file_name[:-4]  # remove .mp4

    re_result = re_file_name.search(video_file_name)  # get video file name
    result_path = f"{'peft' if use_peft else 'audiocraft'}_{dataset}_{musicgen_size}_{re_result.group(1)}"
    audio_result_path = f"{result_dir}/tmp.wav"
    video_result_path = f"{result_dir}/{result_path}_video.mp4"

    gen_audio = torch.squeeze(gen_audio.detach().cpu())  # remove mini-batch dimension, move to CPU for saving
    sample_rate = musicgen_model.sample_rate
    torchaudio.save(audio_result_path, gen_audio, sample_rate)
    audio_file_clip = AudioFileClip(audio_result_path)
    video.audio = audio_file_clip

    print("Rendering Video.")
    video.write_videofile(video_result_path)

    return video_result_path
