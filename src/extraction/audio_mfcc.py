import os

import librosa
import numpy as np

from .split import split_audio

VALID_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')


def extract_mfcc_features(
    audio_path: str,
    n_mfcc: int = 13,
    window_ms: int = 100,
    hop_ms: int = 50,
) -> np.ndarray:
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Formato no soportado: '{ext}'. Use: {VALID_EXTENSIONS}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    frames = split_audio(audio_path, window_ms=window_ms, hop_ms=hop_ms)
    sr = librosa.get_samplerate(audio_path)
    hop_length = int(sr * hop_ms / 1000)

    all_mfcc = []
    for frame in frames:
        mfcc = librosa.feature.mfcc(
            y=frame, sr=sr,
            n_mfcc=n_mfcc,
            n_fft=len(frame),
            hop_length=hop_length,
        )
        all_mfcc.append(mfcc.T)

    return np.vstack(all_mfcc).astype(np.float32) if all_mfcc else np.empty((0, n_mfcc), dtype=np.float32)


__all__ = [
    "extract_mfcc_features",
]
