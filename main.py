import base64
import io
import modal
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from pydantic import BaseModel

from model import AudioCNN

app = modal.App("audio-cnn-inference")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["ffmpeg", "libsndfile1"])  # MP3 support
    .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model")

TARGET_SR = 22050
class AudioProcessor:
    def __init__(self):
        self.sample_rate = TARGET_SR

        self.mel = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025,
        )

        self.db = T.AmplitudeToDB()

    def load_audio(self, audio_bytes):
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != TARGET_SR:
            resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        return waveform

    def extract_features(self, waveform):
        mel = self.mel(waveform)
        mel_db = self.db(mel)
        return mel_db

class InferenceRequest(BaseModel):
    audio_data: str
@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": model_volume},
    scaledown_window=15,
)
class AudioClassifier:

   
    @modal.enter()
    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            "/models/best_model.pth",
            map_location=self.device,
        )

        self.classes = checkpoint["classes"]

        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.processor = AudioProcessor()

        print("✅ Model loaded & ready")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):

        # 1️⃣ Decode audio
        audio_bytes = base64.b64decode(request.audio_data)

        # 2️⃣ Load waveform
        waveform = self.processor.load_audio(audio_bytes)
        waveform_np = waveform.squeeze().cpu().numpy()

        # 3️⃣ Extract features
        features = self.processor.extract_features(waveform)
        spectrogram_np = features.squeeze().cpu().numpy()

        # Add batch dimension
        features = features.unsqueeze(0).to(self.device)

        # 4️⃣ Capture convolution outputs
        visualization = {}
        hooks = []

        def get_hook(name):
            def hook(module, input, output):
                visualization[name] = {
                    "shape": list(output.shape),
                    "values": output[0].detach().cpu().numpy().tolist(),
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(get_hook(name)))

        # 5️⃣ Run model
        with torch.no_grad():
            outputs = self.model(features)
            probs = torch.softmax(outputs, dim=1)[0]
            top3_probs, top3_idx = torch.topk(probs, 3)

        # Remove hooks
        for h in hooks:
            h.remove()

        # 6️⃣ Format predictions
        predictions = [
            {
                "class": self.classes[idx.item()],
                "confidence": prob.item(),
            }
            for prob, idx in zip(top3_probs, top3_idx)
        ]

        # 7️⃣ Final Response (MATCHES FRONTEND)
        return {
            "predictions": predictions,
            "visualization": visualization,
            "input_spectrogram": {
                "shape": list(spectrogram_np.shape),
                "values": spectrogram_np.tolist(),
            },
            "waveform": {
                "values": waveform_np.tolist(),
                "sample_rate": self.processor.sample_rate,
                "duration": float(len(waveform_np) / self.processor.sample_rate),
            },
        }


@app.local_entrypoint()
def main():
    import soundfile as sf
    import requests

    audio_path = "chirpingbirds.wav"

    audio, sr = sf.read(audio_path)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV")

    payload = {
        "audio_data": base64.b64encode(buffer.getvalue()).decode()
    }

    server = AudioClassifier()
    url = server.inference.get_web_url()

    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()

    print("\nTop predictions:")
    for pred in result["predictions"]:
        print(f"  - {pred['class']} {pred['confidence']:.2%}")
