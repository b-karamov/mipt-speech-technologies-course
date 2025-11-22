from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class MelSpectrogramResult:
    """Контейнер с мел-спектрограммой и промежуточными данными."""

    mel_spectrogram: np.ndarray
    sample_rate: int
    hop_samples: int
    window_samples: int
    n_fft: int
    times: np.ndarray
    frequencies: np.ndarray
    mel_frequencies: np.ndarray
    power_spectrogram: np.ndarray
    filter_bank: np.ndarray

def load_audio(path: str, mono: bool = True) -> tuple[np.ndarray, int]:
    """Считываем дорожку с диска и по желанию переводим её в моно."""

    data, sample_rate = sf.read(path)
    if mono and data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float64), int(sample_rate)


def split_track_into_frames(track, L, H):
    """
    track: 1D (N,) звуковая дорожка (N - количество семплов)
    L:     int длина окна в семплах
    H:     int шаг между окнами
    return: 2D (n_frames, L) numpy-массив окон (S, n_frames), где n_frames - количество окон
    """
    track = np.asarray(track)
    N = track.size

    n_frames = int(np.ceil((N - L) / H)) + 1       # количество окон
    n_pad = n_frames * H + L - N                   # падинг для последнего окна

    x = np.pad(track, (0, n_pad), mode="constant") # дорожка с падингом
    s = np.array([x[i * H: i * H + L] for i in range(n_frames)])

    return s, n_frames


def apply_hann_window(frames, window_length):
    """Применяем окно Ханна ко всем фреймам.
    frames: np.ndarray формы (n_frames, window_length)
    return: (window, windowed_frames)
    """
    m = np.arange(window_length)
    window = (1 - np.cos(2 * np.pi * m / (max(window_length - 1, 1)))) / 2
    return window, frames * window


def compute_stft(frames, n_fft, sample_rate):
    """Рассчитываем спектр, частоты, амплитуду и мощность для каждого окна."""
    spectrogram = np.array([np.fft.rfft(frame, n=n_fft) for frame in frames])
    freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
    amplitude = np.abs(spectrogram)
    power = (amplitude ** 2) / n_fft
    return spectrogram, freqs, amplitude, power


def hz_to_mel(f):
    return 1127.0 * np.log(1.0 + (f / 700.0))


def mel_to_hz(m):
    return 700.0 * (np.exp(m / 1127.0) - 1.0)


def build_mel_filterbank(freq_hz, M=40, fmin=50.0, fmax=None, norm="slaney"):
    """
    freq_hz: 1D массив частотных бинов (Гц) длины K (например, np.fft.rfftfreq(...))
    M:       число mel-фильтров
    fmin:    нижняя граница в Гц
    fmax:    верхняя граница в Гц (по умолчанию последняя частота из freq_hz)
    norm:    "htk" (вершина=1) или "slaney" (площадь=1)
    return:  H, матрица формы (M, K)
    """
    K = freq_hz.size
    if fmax is None:
        fmax = freq_hz.max()

    # 1. границы в mel
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)

    # 2. равномерные узлы на mel: M+2 точек (включая края)
    m_points = np.linspace(m_min, m_max, M + 2)

    # 3. обратно в Гц: f_0 ... f_{M+1}
    f_points = mel_to_hz(m_points)

    # 4. для удобства: три сегмента на каждый фильтр i=1..M
    H = np.zeros((M, K), dtype=np.float64)

    for i in range(1, M + 1):
        f_left  = f_points[i - 1]
        f_center= f_points[i]
        f_right = f_points[i + 1]

        # возрастающая часть (f_left -> f_center)
        left_mask = (freq_hz >= f_left) & (freq_hz <= f_center)
        H[i - 1, left_mask] = (freq_hz[left_mask] - f_left) / (f_center - f_left)

        # убывающая часть (f_center -> f_right)
        right_mask = (freq_hz >= f_center) & (freq_hz <= f_right)
        H[i - 1, right_mask] = (f_right - freq_hz[right_mask]) / (f_right - f_center)

    # 5. нормализация
    if norm.lower() == "htk":
        # ничего не делаем: вершина каждого треугольника = 1
        pass
    elif norm.lower() == "slaney":
        # масштабируем так, чтобы площадь каждого треугольника ~ 1
        # дискретная площадь ~ сумма по частотам с учетом шага по частоте
        df = np.mean(np.diff(freq_hz))  # шаг по частоте
        area = H.sum(axis=1, keepdims=True) * df / 2
        H = H * (2.0 / np.maximum(area, 1e-12))
    else:
        raise ValueError("norm must be 'htk' or 'slaney'")

    return H


def mel_spectrogram_from_power(power, freq_hz, M=40, fmin=50.0, fmax=None, 
                               norm="slaney", log=True, eps=1e-10):
    """
    power:   1D (K,) или 2D (K, N) спектр мощности |X|^2/N_FFT
             K — число частотных бинов, N — число окон
    freq_hz: 1D (K,) частоты в герцах, соответствующие power по оси K
    return:  S_mel (M, N) или (M,) если вход был (K,)
    """
    P = np.asarray(power)
    if P.ndim == 1:
        P = P[:, None]  # приведём к (K, 1) для унификации

    H = build_mel_filterbank(freq_hz, M=M, fmin=fmin, fmax=fmax, norm=norm)  # (M, K)
    S_mel = H @ P  # (M, N)

    if log:
        S_mel = np.log(S_mel + eps)

    return S_mel.squeeze(), H


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 60,
    f_min: float = 30.0,
    f_max: float = 8000.0,
    norm: str = "slaney",
    log_scale: bool = True,
    eps: float = 1e-12,
) -> MelSpectrogramResult:
    """Строим mel-спектрограмму для массива аудио в один вызов."""

    window_samples = int(round(window_ms * sample_rate / 1000.0))
    hop_samples = int(round(hop_ms * sample_rate / 1000.0))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_ms and hop_ms must be positive")

    fft_size = 2 ** (int(np.log2(window_samples)) + 1)
    frames, _ = split_track_into_frames(audio, window_samples, hop_samples)
    _, windowed = apply_hann_window(frames, window_samples)

    spectrogram, freqs, amplitude, power = compute_stft(windowed, fft_size, sample_rate)

    mel_spec, filter_bank = mel_spectrogram_from_power(
        power.T, freqs, M=n_mels, fmin=f_min, fmax=f_max, norm=norm, log=log_scale, eps=eps
    )

    effective_fmax = f_max if f_max is not None else freqs.max()
    mel_freqs = mel_to_hz(
        np.linspace(hz_to_mel(f_min), hz_to_mel(effective_fmax), n_mels + 2)
    )[1:-1]

    times = np.arange(mel_spec.shape[1]) * hop_samples / sample_rate

    return MelSpectrogramResult(
        mel_spectrogram=mel_spec,
        sample_rate=sample_rate,
        hop_samples=hop_samples,
        window_samples=window_samples,
        n_fft=fft_size,
        times=times,
        frequencies=freqs,
        mel_frequencies=mel_freqs,
        power_spectrogram=power.T,
        filter_bank=filter_bank,
    )


def compute_mel_spectrogram_from_file(
    audio_path: str,
    mono: bool = True,
    **kwargs,
) -> MelSpectrogramResult:
    """Читаем файл и сразу запускаем вычисление mel-спектрограммы."""

    audio, sample_rate = load_audio(audio_path, mono=mono)
    return compute_mel_spectrogram(audio, sample_rate, **kwargs)


__all__ = [
    "MelSpectrogramResult",
    "apply_hann_window",
    "build_mel_filterbank",
    "compute_stft",
    "compute_mel_spectrogram",
    "compute_mel_spectrogram_from_file",
    "hz_to_mel",
    "load_audio",
    "mel_to_hz",
    "mel_spectrogram_from_power",
    "split_track_into_frames",
]
