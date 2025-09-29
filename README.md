# txtsrtmake_gemma2

This repository demonstrates a simple script for capturing system audio
(through the WASAPI loopback feature) and sending recognized text to
DeepSeek for translation.

## Usage
1. Install the required packages:
   ```bash
   pip install pyaudiowpatch faster-whisper==1.0.1 ctranslate2 numpy librosa \
                python-dotenv openai
   ```
2. Create a `.env` file in the project folder containing your API key.  The
   file should contain lines in the form `KEY=value` with **no quotes** or
   extra characters:
   ```
   DEEPSEEK_API_KEY=sk-...
   # optionally choose which audio device to use
   LOOPBACK_DEVICE_INDEX=0
   ```
   Check carefully that there are no stray characters or spaces – otherwise
   `python-dotenv` will show parse errors.
3. Run the script:
   ```bash
   python zoomtranslatewhisper.py
   ```

If the loopback device cannot be opened, the script will list available
audio devices to help with troubleshooting.
