import os
import re
import sys
import time
import shutil
import threading
import subprocess
import numpy as np
from PIL import Image
from fpdf import FPDF

# ================= CONFIGURATION =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "input")
OUT_DIR = os.path.join(BASE_DIR, "output")
WHISPER_EXE = os.path.join(BASE_DIR, "whisper", "whisper-cli.exe")
MODEL_PATH = os.path.join(BASE_DIR, "whisper", "ggml-large-v3-turbo.bin")

LANGUAGE = "auto"        # Transcription language ("pl", "en", "auto", etc.)
SAMPLE_INTERVAL = 2.0    # Screen sampling interval in seconds
DIFF_THRESHOLD = 0.35    # Pixel diff sensitivity inside the presentation area
SUPPORTED_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
# =================================================

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

LANG_NAMES = {
    "pl": "Polish", "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "uk": "Ukrainian", "ru": "Russian",
    "cs": "Czech", "sk": "Slovak", "nl": "Dutch", "pt": "Portuguese",
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean"
}


def format_lang_label(code):
    code_clean = code.strip().lower()
    name = LANG_NAMES.get(code_clean, code_clean.upper())
    return f"{name} ({code_clean})"


def has_cuda_binaries():
    """Checks for CUDA/cuBLAS DLLs in the whisper directory."""
    whisper_dir = os.path.dirname(WHISPER_EXE)
    if not os.path.exists(whisper_dir):
        return False
    return any("cuda" in f.lower() or "cublas" in f.lower() for f in os.listdir(whisper_dir))


def detect_nvidia_gpu():
    """Queries nvidia-smi to check if an NVIDIA GPU is present and returns its name."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def check_hardware_setup():
    """Verifies whether the installed whisper.cpp build matches the system GPU hardware."""
    cuda_build = has_cuda_binaries()
    gpu_name = detect_nvidia_gpu()

    print("------------------------------------------------------------")
    if gpu_name and cuda_build:
        print(f"[HARDWARE OK] Detected {gpu_name} + CUDA Whisper build.")
        print("              Running with full GPU hardware acceleration.")
        return True
    elif gpu_name and not cuda_build:
        print(f"[WARNING] Detected {gpu_name}, but you are using the CPU-only Whisper build!")
        print("          Transcription will run slowly on your CPU instead of your GPU.")
        print("          Tip: Download 'whisper-bin-win-cuda-12.4.0-x64.zip' into 'whisper/' for 10x-20x speedup.")
        return False
    elif not gpu_name and cuda_build:
        print("[WARNING] CUDA Whisper build detected, but no NVIDIA GPU was found in this system!")
        print("          If whisper-cli fails to start, replace 'whisper/' files with 'whisper-bin-x64.zip' (CPU build).")
        return False
    else:
        print("[HARDWARE INFO] No dedicated NVIDIA GPU detected (integrated/CPU mode).")
        print("                Using CPU Whisper build (optimal for laptops without NVIDIA GPUs).")
        return False


def get_video_duration(video_path):
    """Reads video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def run_with_spinner(message, est_seconds, cmd):
    """Runs a subprocess with a lightweight CLI spinner and elapsed/estimated time."""
    stop_event = threading.Event()
    start_time = time.time()
    est_str = format_duration(est_seconds) if est_seconds > 0 else "a few sec"

    def spin():
        chars = ["|", "/", "-", "\\"]
        idx = 0
        while not stop_event.is_set():
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f"\r  {chars[idx % 4]} {message} [elapsed: {mins:02d}:{secs:02d} | est: ~{est_str}]   ")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.25)

    spinner_thread = threading.Thread(target=spin, daemon=True)
    spinner_thread.start()

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        stop_event.set()
        spinner_thread.join()

    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    sys.stdout.write(f"\r  [OK] {message} (completed in {mins:02d}:{secs:02d})                                        \n")
    sys.stdout.flush()


def run_whisper_with_progress(message, est_seconds, cmd):
    """Runs whisper-cli with -pp flag, prints detected language, and parses live percentage & ETA."""
    start_time = time.time()
    stop_event = threading.Event()
    state = {"pct": 0, "eta": f"~{format_duration(est_seconds)}", "detected_lang": None, "printed_lang": False}

    full_cmd = cmd + ["-pp"]
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    def read_progress():
        buf = b""
        while True:
            ch = proc.stderr.read(1)
            if not ch:
                break
            if ch in (b"\n", b"\r"):
                line = buf.decode("utf-8", errors="ignore")
                buf = b""

                lang_match = re.search(r"auto-detected language:\s*([a-zA-Z-]+)", line)
                if lang_match:
                    state["detected_lang"] = format_lang_label(lang_match.group(1))

                match = re.search(r"progress\s*=\s*(\d+)%", line)
                if match:
                    pct = int(match.group(1))
                    state["pct"] = pct
                    elapsed = time.time() - start_time
                    if pct >= 5:
                        total_pred = elapsed / (pct / 100.0)
                        rem = max(0, total_pred - elapsed)
                        state["eta"] = format_duration(rem)
            else:
                buf += ch

    def render_ui():
        chars = ["|", "/", "-", "\\"]
        idx = 0
        while not stop_event.is_set():
            if state["detected_lang"] and not state["printed_lang"]:
                sys.stdout.write(f"\r  [DETECTED LANGUAGE]: {state['detected_lang']}                                      \n")
                sys.stdout.flush()
                state["printed_lang"] = True

            elapsed_int = int(time.time() - start_time)
            mins, secs = divmod(elapsed_int, 60)
            sys.stdout.write(
                f"\r  {chars[idx % 4]} {message} [{state['pct']}% | elapsed: {mins:02d}:{secs:02d} | ETA: {state['eta']}]     "
            )
            sys.stdout.flush()
            idx += 1
            time.sleep(0.25)

    t_reader = threading.Thread(target=read_progress, daemon=True)
    t_ui = threading.Thread(target=render_ui, daemon=True)
    t_reader.start()
    t_ui.start()

    proc.wait()
    t_reader.join(timeout=1.0)
    stop_event.set()
    t_ui.join()

    if state["detected_lang"] and not state["printed_lang"]:
        sys.stdout.write(f"\r  [DETECTED LANGUAGE]: {state['detected_lang']}                                      \n")

    elapsed_int = int(time.time() - start_time)
    mins, secs = divmod(elapsed_int, 60)
    sys.stdout.write(f"\r  [OK] {message} [100% | completed in {mins:02d}:{secs:02d}]                                         \n")
    sys.stdout.flush()


def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def srt_time_to_seconds(srt_time):
    h, m, s_ms = srt_time.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    segments = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            times = lines[1].split(" --> ")
            start_sec = srt_time_to_seconds(times[0].strip())
            end_sec = srt_time_to_seconds(times[1].strip())
            text = " ".join(line.strip() for line in lines[2:])
            segments.append({"start": start_sec, "end": end_sec, "text": text})
    return segments


def get_presentation_array(img_path):
    """Crops the central presentation area (ignoring Teams sidebar and OS taskbar) for pixel comparison."""
    with Image.open(img_path) as img:
        w, h = img.size
        cropped = img.crop((int(w * 0.02), int(h * 0.08), int(w * 0.82), int(h * 0.92)))
        small = cropped.resize((320, 180)).convert("RGB")
        return np.asarray(small, dtype=np.float32)


def extract_smart_keyframes(video_path, frames_dir, duration):
    """Samples frames at fixed intervals and retains unique slides and final whiteboard drawings."""
    os.makedirs(frames_dir, exist_ok=True)
    raw_pattern = os.path.join(frames_dir, "raw_%05d.jpg")

    est_ffmpeg = max(2, duration / 30.0)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{SAMPLE_INTERVAL},scale=1280:-1",
        "-q:v", "4",
        raw_pattern
    ]
    run_with_spinner("Stage 1/4: Extracting video frames for visual diff...", est_ffmpeg, cmd)

    raw_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("raw_") and f.endswith(".jpg")])
    if not raw_files:
        return []

    kept_frames = []
    prev_arr = None
    prev_path = None
    prev_time = 0.0
    is_drawing_in_progress = False
    total_raw = len(raw_files)

    for idx, fname in enumerate(raw_files):
        if idx % 10 == 0 or idx == total_raw - 1:
            pct = int(((idx + 1) / total_raw) * 100)
            sys.stdout.write(f"\r  / Analyzing visual changes: {pct}% ({idx + 1}/{total_raw} frames)   ")
            sys.stdout.flush()

        curr_time = idx * SAMPLE_INTERVAL
        curr_path = os.path.join(frames_dir, fname)
        curr_arr = get_presentation_array(curr_path)

        if prev_arr is None:
            final_path = os.path.join(frames_dir, f"kept_{idx:05d}.jpg")
            os.replace(curr_path, final_path)
            kept_frames.append({"time": curr_time, "path": final_path})
            prev_arr = curr_arr
            continue

        diff = np.mean(np.abs(curr_arr - prev_arr))

        if diff > 8.0:
            if is_drawing_in_progress and prev_path and os.path.exists(prev_path):
                pre_change_path = os.path.join(frames_dir, f"kept_{idx-1:05d}_final.jpg")
                os.replace(prev_path, pre_change_path)
                kept_frames.append({"time": prev_time, "path": pre_change_path})

            final_path = os.path.join(frames_dir, f"kept_{idx:05d}.jpg")
            os.replace(curr_path, final_path)
            kept_frames.append({"time": curr_time, "path": final_path})
            prev_arr = curr_arr
            prev_path = None
            is_drawing_in_progress = False

        elif diff > DIFF_THRESHOLD:
            if prev_path and os.path.exists(prev_path):
                os.remove(prev_path)
            prev_path = curr_path
            prev_time = curr_time
            prev_arr = curr_arr
            is_drawing_in_progress = True
        else:
            os.remove(curr_path)

    if is_drawing_in_progress and prev_path and os.path.exists(prev_path):
        final_path = os.path.join(frames_dir, "kept_last.jpg")
        os.replace(prev_path, final_path)
        kept_frames.append({"time": prev_time, "path": final_path})

    sys.stdout.write(f"\r  [OK] Extracted {len(kept_frames)} unique slides/drawings.                         \n")
    sys.stdout.flush()
    return kept_frames


def generate_pdf(title, timeline_items, pdf_path):
    """Generates a lightweight PDF combining chronological transcript segments and keyframes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = r"C:\Windows\Fonts\arial.ttf"
    font_bold_path = r"C:\Windows\Fonts\arialbd.ttf"
    if os.path.exists(font_path):
        pdf.add_font("CustomUnicode", "", font_path)
        pdf.add_font("CustomUnicode", "B", font_bold_path if os.path.exists(font_bold_path) else font_path)
        font_family = "CustomUnicode"
    else:
        font_family = "Helvetica"

    pdf.set_font(font_family, "B", 14)
    pdf.cell(0, 10, f"Lecture Notes: {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for item in timeline_items:
        if item["type"] == "text":
            pdf.set_font(font_family, "B", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.write(5, f"[{format_timestamp(item['time'])}] ")

            pdf.set_font(font_family, "", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.write(5, item["content"] + "\n")
            pdf.ln(1)

        elif item["type"] == "image":
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font(font_family, "B", 8)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 5, f"--- Screen Capture [{format_timestamp(item['time'])}] ---", new_x="LMARGIN", new_y="NEXT")
            pdf.image(item["path"], w=170)
            pdf.ln(4)

    pdf.output(pdf_path)


def collect_videos(root_dir):
    """Recursively scans the input directory and preserves relative folder hierarchy."""
    video_tasks = []
    for current_root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTS):
                full_path = os.path.join(current_root, filename)
                rel_dir = os.path.relpath(current_root, root_dir)
                video_tasks.append((full_path, rel_dir, filename))
    return video_tasks


def cleanup_temp(frames_dir, temp_wav, srt_path):
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    if os.path.exists(srt_path):
        os.remove(srt_path)
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)


def process_all():
    if not os.path.exists(WHISPER_EXE) or not os.path.exists(MODEL_PATH):
        print("ERROR: Missing whisper-cli.exe or model file inside the 'whisper/' directory.")
        print("Please check README.md for setup instructions.")
        return

    tasks = collect_videos(VIDEO_DIR)
    if not tasks:
        print(f"No video files found in '{VIDEO_DIR}' (or its subdirectories).")
        return

    gpu_mode = check_hardware_setup()
    if LANGUAGE.strip().lower() == "auto":
        print("[LANGUAGE MODE] Auto-detect (detected per video from first 30s)")
    else:
        print(f"[LANGUAGE MODE] Fixed: {format_lang_label(LANGUAGE)}")
    print("------------------------------------------------------------")
    print(f"Queued {len(tasks)} video file(s) for sequential processing.")

    for idx, (video_path, rel_dir, filename) in enumerate(tasks, 1):
        base_name = os.path.splitext(filename)[0]
        target_out_dir = OUT_DIR if rel_dir == "." else os.path.join(OUT_DIR, rel_dir)
        os.makedirs(target_out_dir, exist_ok=True)

        display_rel_path = filename if rel_dir == "." else os.path.join(rel_dir, filename)
        frames_dir = os.path.join(target_out_dir, f"_temp_{base_name}")
        temp_wav = os.path.join(target_out_dir, f"_temp_{base_name}.wav")
        out_prefix = os.path.join(target_out_dir, f"_temp_{base_name}_sub")
        srt_path = f"{out_prefix}.srt"

        cleanup_temp(frames_dir, temp_wav, srt_path)

        duration = get_video_duration(video_path)
        est_whisper = max(5, duration / 25.0) if gpu_mode else max(10, duration / 3.0)
        est_total = est_whisper + max(3, duration / 25.0)

        print(f"\n============================================================")
        print(f"[{idx}/{len(tasks)}] File: {display_rel_path} (duration: {format_duration(duration)})")
        print(f"Estimated total processing time: ~{format_duration(est_total)}")
        print(f"============================================================")

        try:
            # Stage 1
            frames = extract_smart_keyframes(video_path, frames_dir, duration)

            # Stage 2
            est_audio = max(1, duration / 60.0)
            run_with_spinner(
                "Stage 2/4: Extracting 16kHz mono audio...",
                est_audio,
                ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav]
            )

            # Stage 3
            run_whisper_with_progress(
                "Stage 3/4: Transcribing speech (whisper.cpp)...",
                est_whisper,
                [WHISPER_EXE, "-m", MODEL_PATH, "-f", temp_wav, "-l", LANGUAGE, "-osrt", "-of", out_prefix]
            )

            if not os.path.exists(srt_path):
                print("  [ERROR] whisper-cli failed to generate subtitles. Skipping file...")
                continue

            segments = parse_srt(srt_path)

            txt_path = os.path.join(target_out_dir, f"{base_name}_transcript.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"FULL TRANSCRIPT: {display_rel_path}\n{'='*50}\n\n")
                for seg in segments:
                    f.write(f"[{format_timestamp(seg['start'])} -> {format_timestamp(seg['end'])}] {seg['text']}\n")

            # Stage 4
            sys.stdout.write("  / Stage 4/4: Building synchronized PDF document...")
            sys.stdout.flush()
            timeline = [{"type": "text", "time": s["start"], "content": s["text"]} for s in segments]
            timeline += [{"type": "image", "time": fr["time"], "path": fr["path"]} for fr in frames]
            timeline.sort(key=lambda x: x["time"])

            pdf_path = os.path.join(target_out_dir, f"{base_name}_notes.pdf")
            generate_pdf(display_rel_path, timeline, pdf_path)
            sys.stdout.write(f"\r  [OK] Stage 4/4: Building synchronized PDF document                         \n")

            out_rel_dir = "output" if rel_dir == "." else os.path.join("output", rel_dir)
            print(f"  >>> DONE! Saved to '{out_rel_dir}':")
            print(f"      1. {os.path.basename(pdf_path)}")
            print(f"      2. {os.path.basename(txt_path)}")

        except Exception as e:
            print(f"\n  [ERROR] Failed processing {display_rel_path}: {e}")
        finally:
            cleanup_temp(frames_dir, temp_wav, srt_path)

    print("\n============================================================")
    print("ALL QUEUED VIDEOS PROCESSED.")
    print("============================================================")


if __name__ == "__main__":
    process_all()