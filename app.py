import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import io
import zipfile
import tempfile
import csv

def extract_features(x, fs):
    if len(x) < 128:
        return None
    n_fft = min(1024, len(x))
    hop_length = max(64, n_fft // 4)
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    mfccs = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=13)
    return {
        "rms": np.mean(librosa.feature.rms(y=x)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=x, sr=fs)),
        "mfccs": np.mean(mfccs, axis=1),
    }

def analisar_audio(nome_arquivo, sig, fs, pasta_saida, k=2.0):
    duracao = len(sig) / fs
    S = np.abs(librosa.stft(sig, n_fft=1024, hop_length=256))

    rms = librosa.feature.rms(y=sig, frame_length=1024, hop_length=256)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=fs, hop_length=256)

    threshold = np.mean(rms) + k * np.std(rms)
    eventos = np.where(rms > threshold)[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    librosa.display.waveshow(sig, sr=fs, ax=axes[0])
    axes[0].set_title("Waveform")

    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   sr=fs, hop_length=256,
                                   x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_ylim(0, 40000)
    axes[1].set_title("Espectrograma")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    axes[2].plot(times, rms_db, label="RMS (dB)")
    axes[2].axhline(20*np.log10(threshold+1e-6),
                    color="r", linestyle="--", label=f"Threshold (k={k})")
    axes[2].legend()
    axes[2].set_title("Envelope RMS em dB")
    axes[2].set_ylabel("Amplitude (dB)")

    saida_img = os.path.join(pasta_saida, f"{nome_arquivo}_analise.png")
    plt.tight_layout()
    plt.savefig(saida_img, dpi=300)
    plt.close()

    resultados = []
    for idx in eventos:
        start = idx * 256
        end = start + 1024
        x = sig[start:end]
        f = extract_features(x, fs)
        if f is not None:
            resultados.append(f)

    nivel_medio_db = float(np.mean(rms_db))
    pico_db = float(np.max(rms_db))

    return {
        "arquivo": nome_arquivo,
        "fs": fs,
        "duracao_s": round(duracao, 2),
        "nivel_medio_db": round(nivel_medio_db, 2),
        "pico_db": round(pico_db, 2),
        "eventos": len(eventos),
        "eventos_validos": len(resultados),
        "centroide_medio_hz": round(np.mean([r["spectral_centroid"] for r in resultados]) if resultados else 0, 2)
    }

def processar_audios(uploaded_files, k=2.0):
    with tempfile.TemporaryDirectory() as tmpdir:
        pasta_saida = os.path.join(tmpdir, "resultados")
        os.makedirs(pasta_saida, exist_ok=True)

        resumo = []
        for file in uploaded_files:
            sig, fs = librosa.load(file, sr=None, mono=True)
            nome_arquivo = os.path.splitext(file.name)[0]
            r = analisar_audio(nome_arquivo, sig, fs, pasta_saida, k=k)
            resumo.append(r)

        csv_path = os.path.join(pasta_saida, "resumo.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["arquivo", "fs", "duracao_s", "nivel_medio_db", "pico_db", "eventos", "eventos_validos", "centroide_medio_hz"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in resumo:
                writer.writerow(r)

        # Zipar todos os resultados
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for root, _, files in os.walk(pasta_saida):
                for f in files:
                    caminho = os.path.join(root, f)
                    zf.write(caminho, os.path.basename(caminho))
        zip_buffer.seek(0)

    return zip_buffer


# ------------------- STREAMLIT APP -------------------

st.set_page_config(page_title="Analisador Ultrassom", page_icon="🔊")

st.image("logo.png", width=600)
st.title("Analisador de Ruídos Ultrassônicos - Baltar Engenharia")

uploaded_files = st.file_uploader("Escolha arquivos WAV", type=["wav"], accept_multiple_files=True)

k = st.slider("Sensibilidade (k no threshold)", 1.0, 4.0, 2.0, 0.5)

if uploaded_files:
    if st.button("Processar"):
        zip_buffer = processar_audios(uploaded_files, k=k)
        st.success("Análise concluída!")
        st.download_button(
            label="📥 Baixar Resultados (ZIP)",
            data=zip_buffer,
            file_name="resultados_ultrassom.zip",
            mime="application/zip"
        )
