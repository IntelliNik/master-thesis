import gradio as gr
import os
import sys
sys.path.insert(1, '..')
import inference
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_background_music(video_path, dataset, use_peft, musicgen_size):
    print(f"Start generating background music for {video_path} with model \"{'peft' if use_peft else 'audiocraft'}_{dataset}_{musicgen_size}\"")

    new_video_path = inference.generate_background_music(
        video_path=video_path,
        dataset=dataset,
        musicgen_size=musicgen_size,
        use_stereo=True,
        use_peft=use_peft,
        musicgen_temperature=1.0,
        musicgen_guidance_scale=3.0,
        top_k_sampling=250,
        device=device
    )
    return gr.Video(new_video_path)


interface = gr.Interface(fn=generate_background_music,
                         inputs=[
                                 gr.Video(
                                          label="video input",
                                          min_length=5,
                                          max_length=20,
                                          sources=['upload'],
                                          show_download_button=True,
                                          include_audio=True
                                          ),
                                 gr.Radio(["nature", "symmv"],
                                          label="Video Encoder Version",
                                          value="nature",
                                          info="Choose one of the available Video Encoders."),
                                 gr.Radio([False, True],
                                          label="Use MusicGen Audio Decoder Model trained with PEFT",
                                          value=False,
                                          info="If set to 'True' the MusicGen Audio Decoder models trained with LoRA "
                                               "(Low Rank Adaptation) are used. If set to 'False', the original "
                                               "MusicGen models are used."),
                                 gr.Radio(["small", "medium", "large"],
                                          label="MusicGen Audio Decoder Size",
                                          value="small",
                                          info="Choose the size of the MusicGen audio decoder."),
                                ],

                         outputs=[gr.Video(label="video output")],
                         examples=[
                             [os.path.abspath("../../../videos/originals/n_1.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_2.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_3.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_4.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_5.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_6.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_7.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/n_8.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_1.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_2.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_3.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_4.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_5.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_6.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_7.mp4"), "nature", True, "small"],
                             [os.path.abspath("../../../videos/originals/s_8.mp4"), "nature", True, "small"],
                           ],
                         cache_examples=False
                         )

if __name__ == "__main__":
    interface.launch(
        share=False
    )
