from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator([model], cfg)

text = "Hello, this is a ceci's test run."

sample = TTSHubInterface.get_model_input(task, text)

sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to("cuda")
model = model.to("cuda")

wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
wav = wav.cpu().numpy()


import soundfile as sf


output_file = "output_fast_speech2.wav"
sf.write(output_file, wav, rate)

print(f"output file saved: {output_file}")