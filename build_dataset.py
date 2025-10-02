import os
import cv2
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extrae frames de un video cada `frame_rate` segundos.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # evita división por 0
        print(f"⚠️ No se pudo leer FPS de {video_path}, saltando...")
        return
    interval = int(fps * frame_rate)  # cada X segundos

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"✅ Extraídos {saved} frames de {video_path}")


def build_dataset(base_dir="dataset", frame_rate=1):
    """
    playlists: dict con {clase: url_playlist}
    """
    video_dir = Path(base_dir) / "video"
    frames_dir = Path(base_dir) / "frame"

    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Descargar TODOS los videos de la playlist
    #download_videos(url, video_dir)

    # Extraer frames de cada video descargado
    for video_file in os.listdir(video_dir):
        if video_file.endswith((".mp4")):
            video_path = video_dir / video_file
            extract_frames(str(video_path), frames_dir, frame_rate)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_rate", type=int, default=2,
                        help="Cada cuantos segundos guardar un frame")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Carpeta base para guardar dataset")
    args = parser.parse_args()

    #video para descarga
    # https://youtu.be/0On2smxRndA?si=CORC33SmbZz1-eSR

    build_dataset(base_dir=args.output_dir, frame_rate=args.frame_rate)
