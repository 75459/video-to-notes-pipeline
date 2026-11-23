
# Video-to-Notes Pipeline (Video to Synchronized PDF & Transcript)

A lightweight, fully local batch-processing utility that converts long video recordings (such as webinars, tutorials, presentations, online meetings, or lectures in `.mp4`, `.mkv`, `.mov`, `.avi`, or `.webm` format) into compact, searchable study and reference materials:

1. **`*_notes.pdf`** — A chronological PDF document combining timestamped speech transcripts with automatically captured slides, code screens, and completed whiteboard drawings.
2. **`*_transcript.txt`** — A complete, verbatim, timestamped text transcript of the entire video.

Instead of manually scrubbing through hours of video or relying on blind fixed-interval screenshots (which miss erased whiteboard sketches or generate dozens of duplicate frames), this pipeline detects actual visual changes in the presentation area and aligns them chronologically with local Whisper speech-to-text output.

---

## Hardware Compatibility (GPU vs. CPU)

This pipeline relies on `whisper.cpp` for local speech recognition. Depending on your computer's graphics card, it operates in one of two modes:

* **NVIDIA GPUs (RTX / GTX — Recommended):** Full hardware acceleration via **CUDA**. Processing a 45-minute video typically takes **15–45 seconds**.
* **AMD / Intel GPUs or Integrated Graphics (CPU Mode):** Prebuilt Windows x64 binaries for `whisper.cpp` do not include out-of-the-box AMD ROCm or Intel GPU backends. If you use an AMD Radeon, Intel Arc, or laptop integrated GPU, **the program runs in CPU mode** on your processor. It works reliably out of the box on any modern Intel/AMD CPU, though transcription takes longer (roughly 1/3 of the video duration).
  > *Technical Note for Advanced Users:* If you own an AMD or Intel GPU and want hardware acceleration, you can compile `whisper.cpp` from source with Vulkan enabled (`-DGGML_VULKAN=1`) and drop the compiled binaries into the `whisper/` folder—the script will automatically use them.

---

## Getting Started (Step-by-Step Guide for Beginners)

Follow these steps in exact order. No prior programming experience is required.

### Step 1: Install Python
1. Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest Windows installer.
2. Run the installer. **CRITICAL:** On the first screen, check the box **"Add python.exe to PATH"** at the bottom before clicking *Install Now*.

### Step 2: Install FFmpeg (Audio & Video Engine)
FFmpeg is required to extract audio tracks and sample video frames.
1. Open Windows **PowerShell** or **Command Prompt** (`cmd`).
2. Paste the following command and press `Enter`:
   ```powershell
   winget install Gyan.FFmpeg
   ```
3. Wait for the installation to finish, then **close and reopen** your terminal window so Windows recognizes the new command.

### Step 3: Download This Project
* **Option A (Recommended for beginners):** Go to the [**Releases**](https://github.com/75459/video-to-pdf-transcript/releases/latest) page, download **`Source code (zip)`** under *Assets*, and extract it to a simple folder path without special characters, e.g., `C:\VideoToNotes\`.
* **Option B (Git):** Clone the repository directly:
  ```bash
  git clone https://github.com/75459/video-to-pdf-transcript.git
  cd video-to-pdf-transcript
  ```

### Step 4: Install Required Python Libraries
1. Open **PowerShell** inside your extracted project folder (you can right-click inside the folder in Windows Explorer and select *Open in Terminal*).
2. Run:
   ```powershell
   pip install -r requirements.txt
   ```

### Step 5: Download the Speech Recognition Engine (`whisper.cpp`)
Because executable binaries and AI models are too large for GitHub, you need to place them into the `whisper/` folder manually:
1. Go to the official [ggml-org/whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases) page.
2. Scroll down to **Assets** of the latest release and download the right archive for your hardware:
   - **If you have an NVIDIA GPU (RTX 20/30/40/50 series or GTX):** Download **`whisper-bin-win-cuda-12.4.0-x64.zip`** (or the latest `cuda-12.x-x64.zip` build).
   - **If you have an AMD GPU, Intel GPU, or no dedicated graphics card:** Download **`whisper-bin-x64.zip`**.
3. Open the downloaded `.zip` archive and extract **`whisper-cli.exe`** along with **all `.dll` files** directly into the **`whisper/`** folder inside your project directory.

### Step 6: Download the Speech Model
1. Go to [Hugging Face — ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main).
2. Download **`ggml-large-v3-turbo.bin`** (~1.6 GB — offers the best balance of speed and multilingual accuracy, including Polish and English).
3. Move the downloaded `ggml-large-v3-turbo.bin` file into the **`whisper/`** folder.

Your `whisper/` folder should now contain `whisper-cli.exe`, several `.dll` files, and `ggml-large-v3-turbo.bin`.

---

## Usage

1. **(Optional) Set Transcription Language:** Open `process_lectures.py` in Notepad and check line 17 (`LANGUAGE = "pl"`). Change `"pl"` to `"en"` for English recordings or `"auto"` for automatic language detection.
2. **Add Videos:** Drop your video files (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`) into the **`input/`** folder.
   * You can place files directly in `input/` or organize them into nested folders (e.g., `input/Module1/session01.mp4`).
3. **Start Processing:**
   * Double-click **`START.bat`** (Windows), or run `python process_lectures.py` in your terminal.
   * The script automatically verifies your GPU/CPU setup, processes all queued videos sequentially, and displays live percentage and ETA progress.
4. **Collect Results:**
   * Open the **`output/`** folder. Your `*_notes.pdf` and `*_transcript.txt` files will be saved there, mirroring the exact subfolder structure from `input/`.

---

## Architecture & Technical Nuances

* **Smart Visual Diff (`FFmpeg` + `Pillow` + `NumPy`):**
  * Samples frames every 2 seconds (`SAMPLE_INTERVAL = 2.0`) and crops the central presentation region (`2%–82%` width, `8%–92%` height) to ignore meeting participant sidebars and OS taskbars so speaker avatar highlights do not trigger false captures.
  * Uses two-tier pixel delta thresholds (`DIFF_THRESHOLD = 0.35` for incremental whiteboard annotations and `> 8.0` for full slide transitions). When a user finishes drawing on a digital whiteboard and switches slides, the pipeline saves the **final completed drawing right before the transition** rather than an empty board.
* **Non-Blocking Progress & ETA (`threading` + `subprocess`):**
  * Captures unbuffered byte streams (`bufsize=0`) from `whisper-cli.exe -pp` on a background reader thread while a separate UI thread updates the CLI timer every `0.25s`, preventing console freezes during heavy inference with near-zero CPU/RAM overhead.
* **Resilient Batch Queue:**
  * Recursively traverses `input/` via `os.walk`, cleans up any leftover `_temp_*` artifacts before and after each file, and isolates exceptions per file so one corrupted video never aborts the rest of the queue.

---

## Development Note
This utility was built to solve a real personal workflow bottleneck—converting dozens of lengthy video recordings into concise, structured, and searchable PDF/text notes without relying on cloud upload limits. The pipeline architecture, frame-diffing logic requirements, and threshold tuning were designed by me, while the Python implementation was generated and iterated with AI assistance (Gemini).
