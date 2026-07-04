"""Dataset downloader for REVIVID.

Reads a list of URLs from ``config/sources.yaml`` and downloads each one.
Supports plain HTTP(S) files (optionally zipped), Google Drive, and YouTube
(via yt-dlp). Downloaded assets are placed under ``data/raw/<section_name>/``.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
import yt_dlp


class DatasetDownloader:
    """Download raw video datasets listed in ``config/sources.yaml``."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.download_dir = self.project_root / "data" / "raw"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_source_file = self.project_root / "config" / "sources.yaml"

    def download_all(self):
        print("Start downloading datasets, this may take a while...")

        yaml_sources = self._read_yaml_file()

        if yaml_sources:
            for section_name, urls in yaml_sources.items():
                if not urls:
                    print(f"Skipping empty or missing section: {section_name}")
                    continue

                section_dir = str(self.download_dir / section_name.lower())
                os.makedirs(section_dir, exist_ok=True)
                print(
                    f"\nProcessing section: {section_name} -> saving to {section_dir}"
                )

                for url in urls:
                    if not url:
                        continue
                    domain = urlparse(url).netloc.lower()
                    if "youtube.com" in domain or "youtu.be" in domain:
                        self._download_youtube_video(url.strip(), section_dir)
                    elif "drive.google.com" in domain:
                        self._download_google_drive_file(url.strip(), section_dir)
                    else:
                        self._download_file(url.strip(), section_dir)

        print("\nFinished downloading all datasets.\n")

    def _read_yaml_file(self) -> dict:
        if not os.path.exists(self.yaml_source_file):
            print(f"Warning: YAML file '{self.yaml_source_file}' not found.")
            return {}
        with open(self.yaml_source_file, "r", encoding="utf-8") as file:
            try:
                data = yaml.safe_load(file)
                return data if data else {}
            except yaml.YAMLError as parse_error:
                print(f"Error parsing YAML file: {parse_error}")
                return {}

    def _create_filename(self, url: str) -> str:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            raise ValueError(f"Cannot derive a filename from URL: {url}")
        _, file_extension = os.path.splitext(filename)
        if not file_extension:
            filename = "unknown_name.zip"
        return filename

    def _extract_zip(self, file_path: str, target_dir: str) -> None:
        if not file_path.lower().endswith(".zip"):
            return
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
        except zipfile.BadZipFile:
            print(f"Failed to extract {file_path}. Invalid or corrupted zip file.")
        os.remove(file_path)

    def _download_file(self, url: str, target_dir: str) -> None:
        filename = self._create_filename(url)
        file_path = f"{target_dir}/{filename}"
        print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded: {filename}")
            self._extract_zip(file_path, target_dir)
        except requests.exceptions.RequestException as req_error:
            print(f"Failed to download {url}. Error: {req_error}")
        except Exception as general_error:
            print(
                f"An unexpected error occurred processing {url}. Error: {general_error}"
            )

    def _download_google_drive_file(self, url: str, target_dir: str) -> None:
        filename = self._create_filename(url)
        file_path = f"{target_dir}/{filename}"
        print(f"Downloading Google Drive file from {url}...")
        try:
            session = requests.Session()
            response = session.get(url, stream=True)
            if "text/html" in response.headers.get("Content-Type", ""):
                file_id = url.split("id=")[-1] if "id=" in url else None
                download_url = "https://drive.usercontent.google.com/download"
                params = {"id": file_id, "export": "download", "confirm": "t"}
                response = session.get(download_url, params=params, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            self._extract_zip(file_path, target_dir)
        except requests.exceptions.RequestException as req_error:
            print(f"Failed to download {url}. Error: {req_error}")
        except Exception as general_error:
            print(
                f"An unexpected error occurred processing {url}. Error: {general_error}"
            )

    def _download_youtube_video(self, url: str, target_dir: str) -> None:
        ydl_opts = {
            "format": "bestvideo[height=1080]+bestaudio/best[height=1080]",
            "outtmpl": f"{target_dir}/%(title)s.%(ext)s",
            "ignoreerrors": True,
            "no_warnings": True,
            "quiet": True,
            "merge_output_format": "mp4",
        }
        delay_seconds = 10
        print(f"Downloading YouTube video from {url}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                if info_dict is None:
                    raise ValueError()
                video_title = info_dict.get("title", "Unknown Title")
                video_extension = info_dict.get("ext", "unknown")
                print(f"Successfully downloaded: {video_title}.{video_extension}")
        except ValueError:
            time.sleep(1)
            print(f"Failed to download video from URL: {url}. Skipping...")
        except Exception:
            print(f"An unexpected error occurred for {url}. Skipping...")
        time.sleep(delay_seconds)
