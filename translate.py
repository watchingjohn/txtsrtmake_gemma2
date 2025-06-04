import os
import time
import re
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd

# Tenacity 관련 import (필수!)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError

# requests import (Ollama 사용)
import requests
import json

# ── Ollama 기반 모델 설정 및 로딩 ──────────────────────────────────────
MODEL_NAME = "gemma3:27b-it-qat"
N_CTX = 128000
CHUNK_SIZE_TOKENS = 4000
CHUNK_OVERLAP_TOKENS = 200
BASE_MAX_TOKENS = 2000
TEMPERATURE = 0.3
FORCE_CHUNK = False

print(f"\ud83d\udd04 Ollama \ubaa8\ub378 \uc5f0\uacb0 \uc2dc\ub3c4: {MODEL_NAME}")
print("\u23f3 \uc7a0\uc2dc\ub9cc \uae30\ub2e4\ub824\uc8fc\uc138\uc694...")

class OllamaModel:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        print(f"\ud83d\udd17 Ollama \ubaa8\ub378 \uc124\uc815: {model_name}")
    
    def __call__(self, prompt, max_tokens=2000, temperature=0.3, stop=None, echo=False):
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stop": ["<end_of_turn>", "</s>"] if stop is None else stop
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return {"choices": [{"text": result.get("response", "")}]}
            else:
                return {"choices": [{"text": f"[Ollama HTTP \uc624\ub958: {response.status_code}]"}]}
                
        except Exception as e:
            return {"choices": [{"text": f"[Ollama \uc5f0\uacb0 \uc624\ub958: {e}]"}]}

try:
    print("\ud83d\udd0d Ollama \uc11c\ube44\uc2a4 \ud655\uc778 \uc911...")
    test_response = requests.get("http://localhost:11434/api/tags", timeout=10)
    
    if test_response.status_code == 200:
        available_models = test_response.json().get("models", [])
        model_names = [m.get("name", "") for m in available_models]
        
        print(f"\ud83d\udccb \uc124\uce58\ub41c \ubaa8\ub378\ub4e4: {model_names}")
        
        gemma3_models = [name for name in model_names if "gemma3" in name.lower()]
        
        if gemma3_models:
            if MODEL_NAME in model_names:
                selected_model = MODEL_NAME
            else:
                selected_model = gemma3_models[0]
            
            print(f"\u2705 \uc0ac\uc6a9\ud560 \ubaa8\ub378: {selected_model}")
            model = OllamaModel(selected_model)
            
        else:
            print("\u26a0\ufe0f  Gemma3 \ubaa8\ub378\uc774 \uc124\uce58\ub418\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4!")
            print("\ub2e4\uc74c \uba85\ub839\uc73c\ub85c \uc124\uce58\ud558\uc138\uc694:")
            print(f"ollama pull {MODEL_NAME}")
            print("\n\ub610\ub294 \ub2e4\ub978 \ubaa8\ub378 \uc0ac\uc6a9:")
            print("ollama pull gemma3:27b")
            
            if model_names:
                fallback_model = model_names[0]
                print(f"\u26a1 \ub300\uc2e0 {fallback_model} \ubaa8\ub378\uc744 \uc0ac\uc6a9\ud569\ub2c8\ub2e4")
                model = OllamaModel(fallback_model)
            else:
                raise Exception("\uc0ac\uc6a9 \uac00\ub2a5\ud55c \ubaa8\ub378\uc774 \uc5c6\uc2b5\ub2c8\ub2e4")
    else:
        raise Exception(f"Ollama \uc11c\ube44\uc2a4 \uc5f0\uacb0 \uc2e4\ud328 (HTTP {test_response.status_code})")
        
except Exception as e:
    print(f"\u274c Ollama \uc5f0\uacb0 \uc2e4\ud328: {e}")
    print("\n\ud574\uacb0 \ubc29\ubc95:")
    print("1. Ollama\uac00 \uc124\uce58\ub418\uc5b4 \uc788\ub294\uc9c0 \ud655\uc778 (https://ollama.ai)")
    print("2. Ollama \uc11c\ube44\uc2a4\uac00 \uc2e4\ud589 \uc911\uc778\uc9c0 \ud655\uc778")
    print("3. \uba85\ub839\uc5b4: ollama serve")
    print("4. \ubaa8\ub378 \uc124\uce58: ollama pull gemma3:27b-it-qat")
    
    class DummyModel:
        def __call__(self, prompt, max_tokens=2000, temperature=0.3, stop=None, echo=False):
            return {"choices": [{"text": "[Ollama \uc5f0\uacb0 \ud544\uc694] \uc704\uc758 \ud574\uacb0\ubc95\uc744 \ub530\ub77c\ud558\uc138\uc694"}]}
    
    model = DummyModel()
    print("\u26a0\ufe0f  \ub354\ubbf8 \ubaa8\ub378 \ubaa8\ub4dc\ub85c \uc2e4\ud589\ub429\ub2c8\ub2e4")

print("=" * 60)

class TextProcessor:
    def count_tokens(self, text):
        korean_chars = len([c for c in text if '\uac00' <= c <= '\ud7af'])
        other_chars = len(text) - korean_chars
        words = len(text.split())
        
        if korean_chars > other_chars:
            return korean_chars * 1.1
        else:
            return words * 1.3
    
    def contains_korean(self, text):
        return any('\uac00' <= char <= '\ud7af' for char in text)
    
    def estimate_output_tokens(self, text, is_translation, multiplier):
        input_tokens = self.count_tokens(text)
        return int(input_tokens * multiplier)
    
    def get_optimal_chunk_size(self, text, n_ctx, overlap):
        return CHUNK_SIZE_TOKENS, overlap
    
    def chunk_text(self, text, chunk_size, overlap):
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + ". "
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def merge_translations(self, original_chunks, translated_chunks, overlap):
        return '\n\n'.join(translated_chunks)

class GPUManager:
    def cleanup(self):
        import gc
        gc.collect()

class Logger:
    def debug(self, msg):
        print(f"DEBUG: {msg}")
    def info(self, msg):
        print(f"INFO: {msg}")
    def warning(self, msg):
        print(f"WARNING: {msg}")
    def error(self, msg, exc_info=None):
        print(f"ERROR: {msg}")

text_processor = TextProcessor()
gpu_manager = GPUManager()
logger = Logger()

print("\ud83d\ude80 Ollama \uae30\ubc18 Gemma3 \ubc88\uc5ed \uc2dc\uc2a4\ud15c \uc900\ube44 \uc644\ub8cc!")
print(f"\ud83e\udd16 \ubaa8\ub378: {MODEL_NAME}")
print(f"\ud83e\udd14 \uc5f4\uae30 \ubcbd: {N_CTX:,} \ud1a0\ud070")
print(f"\ud83c\udf21\ufe0f  \uc628\ub3c4: {TEMPERATURE}")
print("=" * 60)

required_libs = {
    "requests": "HTTP \ud1b5\uc2e0\uc6a9",
    "pandas": "Excel \ud30c\uc77c \uc77d\uae30\uc6a9", 
    "tqdm": "\uc9c4\ud589\b82c \ud45c\uc2dc\uc6a9"
}

missing_libs = []
for lib, desc in required_libs.items():
    try:
        __import__(lib)
        print(f"\u2705 {lib} \uc124\uce58\ub428 ({desc})")
    except ImportError:
        print(f"\u274c {lib} \uc124\uce58 \ud544\uc694 ({desc})")
        missing_libs.append(lib)

if missing_libs:
    print("\n\ub2e4\uc74c \uba85\ub839\uc73c\ub85c \uc124\uce58\ud558\uc138\uc694:")
    print(f"pip install {' '.join(missing_libs)}")
    if "pandas" in missing_libs:
        print("Excel \ud30c\uc77c \uc9c0\uc6d0: pip install openpyxl")
    exit(1)


def find_sentence_end_near_line(lines: List[str], target_line: int, search_range: int = 10) -> int:
    sentence_endings = ['.', '!', '?', '\u3002', '\uff01', '\uff1f']
    search_start = max(0, target_line - search_range)
    search_end = min(len(lines), target_line + search_range)
    for i in range(target_line, search_end):
        if i < len(lines):
            line = lines[i].strip()
            if any(line.endswith(ending) for ending in sentence_endings):
                logger.debug(f"Found sentence end at line {i+1}: {line[:50]}...")
                return i + 1
    for i in range(target_line, search_start, -1):
        if i < len(lines):
            line = lines[i].strip()
            if any(line.endswith(ending) for ending in sentence_endings):
                logger.debug(f"Found sentence end at line {i+1}: {line[:50]}...")
                return i + 1
    logger.warning(f"No sentence end found near line {target_line}, using target line")
    return target_line

def split_txt_content(content: str, lines_per_chunk: int = 300) -> List[str]:
    lines = content.split('\n')
    chunks = []
    current_start = 0
    logger.info(f"Splitting TXT content: {len(lines)} lines into ~{lines_per_chunk} line chunks")
    while current_start < len(lines):
        target_line = current_start + lines_per_chunk
        if target_line >= len(lines):
            chunk_lines = lines[current_start:]
            chunk_text = '\n'.join(chunk_lines)
            if chunk_text.strip():
                chunks.append(chunk_text)
            break
        split_point = find_sentence_end_near_line(lines, target_line)
        chunk_lines = lines[current_start:split_point]
        chunk_text = '\n'.join(chunk_lines)
        if chunk_text.strip():
            chunks.append(chunk_text)
        current_start = split_point
    logger.info(f"TXT content split into {len(chunks)} chunks")
    return chunks

def parse_srt_entry(entry_text: str) -> dict:
    lines = entry_text.strip().split('\n')
    if len(lines) < 3:
        return None
    try:
        entry_number = int(lines[0])
        timestamp = lines[1]
        subtitle_text = '\n'.join(lines[2:])
        return {
            'number': entry_number,
            'timestamp': timestamp,
            'text': subtitle_text,
            'original': entry_text
        }
    except (ValueError, IndexError):
        logger.warning(f"Failed to parse SRT entry: {entry_text[:100]}...")
        return None

def split_srt_content(content: str, entries_per_chunk: int = 300) -> List[str]:
    entries = re.split(r'\n\s*\n', content.strip())
    entries = [entry for entry in entries if entry.strip()]
    logger.info(f"Splitting SRT content: {len(entries)} entries into ~{entries_per_chunk} entry chunks")
    chunks = []
    for i in range(0, len(entries), entries_per_chunk):
        chunk_entries = entries[i:i + entries_per_chunk]
        chunk_content = '\n\n'.join(chunk_entries)
        if chunk_content.strip():
            chunks.append(chunk_content)
    logger.info(f"SRT content split into {len(chunks)} chunks")
    return chunks

def create_translation_prompt(text: str, source_lang: str = "\uc601\uc5b4", target_lang: str = "\ud55c\uad6d\uc5b4") -> str:
    prompt = f"""<bos><start_of_turn>user
\uc544\ub798 \ud14d\uc2a4\ud2b8\ub294 \ubd88\uad50\uc5d0 \uad00\uc2ec\uc774 \ub9ce\uc740 \uc77c\ubc18 \ub300\uc911\uc774 \ud568\uaed8 \uc77d\ub294 \uc790\ub8cc\ub85c \ud65c\uc6a9\ub420 \uc608\uc815\uc774\uc57c.
\ud14d\uc2a4\ud2b8\uc5d0\ub294 \ud2f0\ubca0\ud2b8 \ubd88\uad50, \ud2f0\ubca0\ud2b8 \uc0cc\ubc14 \uae4c\uaddc(Shangpa Kagyu) \uad00\ub828\ub41c \uc804\ubb38 \uc6a9\uc5b4\uac00 \ub2e4\uc218 \ud3ec\ud568\ub418\uc5b4 \uc788\uc5b4.
\ub2e4\uc74c \uae30\uc900\uc5d0 \ub530\ub77c \ubaa8\ub4e0 \ub0b4\uc6a9\uc744 \ube68\uc9c0\uc9c0 \uc54a\uac8c \uc720\uc9c0\ud558\uba70 \uacf5\uc18c\ud558\uace0, \uc804\ubb38\uc801\uc774\uba70\ub3c4 \uc27d\uac8c \uc77d\ud788\ub294 \ub300\uc911 \ucc3d\ud56d\uc801 \ubb38\uccb4\ub85c \uc719\ubb38\ud574\uc918.

\ud544\uc218 \uc870\uac74:
- \ud398\uc774\uc9c0 \ubc84\uc804\uc744 \uad6c\ubd84\ud558\uace0, \ud398\uc774\uc9c0\ub9c8\ub2e4 \ucda9\ubd84\ud788 \uacf5\ubc31\uc744 \ub2e4\ub0b4\uc11c \ud398\uc774\uc9c0\uc640 \ud398\uc774\uc9c0\uac00 \ud655\uc2e4\ud788 \uad6c\ubd84\ub418\ub3c4\ub85d \ud574\uc918.
- \ub0b4\uc6a9 \ub204\ub77d \uc5c6\uc774 \uc6d0\ubb38\uc5d0 \ud3ec\ud568\ub41c \ubaa8\ub4e0 \uc815\ubcf4\ub97c \ub2e4 \ub2f5\uc544\uc918.
- \ud2f0\ubca0\ud2b8\uc5b4 \ubc0f \ubd88\uad50 \uc804\ubb38 \uc6a9\uc5b4\ub294 \uac00\ub2a5\ud55c \ud55c \uadf8\ub8f9\uad6c\ub098 \uacac\ud55c \uc124\uba85\uc744 \ucd94\uac00\ud574\uc918.
- \uc804\uccb4 \ubb38\uc7a5\uc740 \uacf5\uc18c\ud558\uace0 \uc804\ubb38\uc801\uc778 \ubb38\uc5b4\ucc44\ub85c \uc815\ub9ac\ud574\uc918. \ud55c\uad6d\uc778\uc774 \uc27d\uace0 \uc790\uc720\uc2a4\ub7fd\uace0 \ud3b8\ud55c\uac83\uc73c\ub85c \uc77d\uc744 \uc218 \uc788\ub3c4\ub85d \uc719\ubb38\ud574\uc918. \uadf8\ub9ac\uace4 \uc784\uc758\ub85c \ub9cc\ub4e4\uc5b4\ub0bc \uc218 \uc788\ub294 \ub0b4\uc6a9\uc758 \ubc88\uc5ed \ubd80\ubd84\uc740 \uc5c6\ub3c4\ub85d \ud574\uc918
- \uadf8\ub9ac\uace4 \uc601\ubb38\uc740 \ud45c\uae30\ud558\uc9c0 \ub9c8\ub9ac\uace0 \ubc88\uc5ed \ubc0f \uc719\ubb38\ub41c \ud55c\uad6d\uc744\ub9cc \ud45c\uae30\ud574\uc918

{source_lang} \uc6d0\ubb38:
{text}
<end_of_turn>
<start_of_turn>model
"""
    
    return prompt

def create_srt_translation_prompt(srt_chunk: str, source_lang: str = "\uc601\uc5b4", target_lang: str = "\ud55c\uad6d\uc5b4") -> str:
    prompt = f"""<bos><start_of_turn>user
\uc544\ub798 \ud14d\uc2a4\ud2b8\ub294 \ubd88\uad50\uc5d0 \uad00\uc2ec\uc774 \ub9ce\uc740 \uc77c\ubc18 \ub300\uc911\uc774 \ud568\uaed8 \uc77d\ub294 \uc790\ub8cc\ub85c \ud65c\uc6a9\ub420 \uc608\uc815\uc774\uc57c.
\ud14d\uc2a4\ud2b8\uc5d0\ub294 \ud2f0\ubca0\ud2b8 \ubd88\uad50, \ud2f0\ubca0\ud2b8 \uc0cc\ubc14 \uae4c\uaddc(Shangpa Kagyu) \uad00\ub828\ub41c \uc804\ubb38 \uc6a9\uc5b4\uac00 \ub2e4\uc218 \ud3ec\ud568\ub418\uc5b4 \uc788\uc5b4.
\ub2e4\uc74c \uae30\uc900\uc5d0 \ub530\ub77c \ubaa8\ub4e0 \ub0b4\uc6a9\uc744 \ube68\uc9c0\uc9c0 \uc54a\uac8c \uc720\uc9c0\ud558\uba70 \uacf5\uc18c\ud558\uace0, \uc804\ubb38\uc801\uc774\uba70\ub3c4 \uc27d\uac8c \uc77d\ud788\ub294 \ub300\uc911 \ucc3d\ud56d\uc801 \ubb38\uccb4\ub85c \uc719\ubb38\ud574\uc918.

\ud544\uc218 \uc870\uac74:
- \ub0b4\uc6a9 \ub204\ub77d \uc5c6\uc774 \uc6d0\ubb38\uc5d0 \ud3ec\ud568\ub41c \ubaa8\ub4e0 \uc815\ubcf4\ub97c \ub2e4 \ub2f5\uc544\uc918.
- \ud2f0\ubca0\ud2b8\uc5b4 \ubc0f \ubd88\uad50 \uc804\ubb38 \uc6a9\uc5b4\ub294 \uac00\ub2a5\ud55c \ud55c \uadf8\ub8f9\uad6c\ub098 \uacac\ud55c \uc124\uba85\uc744 \ucd94\uac00\ud574\uc918.
- \uc804\uccb4 \ubb38\uc7a5\uc740 \uacf5\uc18c\ud558\uace0 \uc804\ubb38\uc801\uc778 \ubb38\uc5b4\ucc44\ub85c \uc815\ub9ac\ud574\uc918. \ud55c\uad6d\uc778\uc774 \uc27d\uace0 \uc790\uc720\uc2a4\ub7fd\uace0 \ud3b8\ud55c\uac83\uc73c\ub85c \uc77d\uc744 \uc218 \uc788\ub3c4\ub85d \uc719\ubb38\ud574\uc918. \uadf8\ub9ac\uace4 \uc784\uc758\ub85c \ub9cc\ub4e4\uc5b4\ub0bc \uc218 \uc788\ub294 \ub0b4\uc6a9\uc758 \ubc88\uc5ed \ubd80\ubd84\uc740 \uc5c6\ub3c4\ub85d \ud574\uc918
- \ud0c0\uc784\uc2a4\ud2b8\ubcfc\uc2a4-\ubb38\uc7a5 \uc77c\uce58 \ud655\uc778: \uac01 \ud0c0\uc784\uc2a4\ud2b8\uc5d0 \ud574\ub2f9\ud558\ub294 \uc6d0\ubcf8 \uc601\ubb38 \ubb38\uc7a5\uc744 \uc815\ud655\ud788 \ud655\uc778\ud558\uace0 \ubc88\uc5ed\uc744 \uc9c4\ud589\ud574\uc918.
- \uc804\ubb38 \uc6a9\uc5b4 \ubc0f \ubb38\ubcf5 \uc7a5\ubcf5: \ud2f0\ubca0\ud2b8\uc5b4 \ubc0f \ubd88\uad50 \uc804\ubb38 \uc6a9\uc5b4\ub294 \uc6d0\uc5b4\uc640 \ud568\uaed8 \uacac\ud55c \uc124\uba85\uc744 \ucd94\uac00\ud558\uace0, \uc804\uccb4\uc801\uc778 \ubb38\ubf1c\uc744 \uace0\ub824\ud574 \uc790\uc720\uc2a4\ub7fd\uace0 \uc815\ud655\ud55c \ubc88\uc5ed\uc744 \ud574\uc918.
- \ub204\ub77d\ub41c \ub0b4\uc6a9 \ubcf4\uc874: \ud638\uc2a4\ub7ec \ub3d9\uc0ac\ub294 \uc9c0\uae08\ub3c4 \uc9d1\uc2dc\uac80 \uc911\uc774\ub77c \uadf8\ub9ac\uace4 \ud544\uc694\ud55c \uc815\ubcf4\ub294 \uac00\uc9c0\ubcf4\uace0 \uc788\ub2e4. \ud55c\uad6d\uc758 \uc9c0\uae08 \ubb38\ud654\ub294 \ub3d9\uc0ac\uc640 \ub300\ub9cc\ud574\uc11c \ub9ce\uc740 \ub178\ub3d9\uc218\uac00 \ubb38\ud654 \uc7ac\ud654\uac00 \ub418\ub294 \ubc18\uba74 \uc804\ubb38 \ubb38\uc5b4\ub97c \uc5bc\uc5b4\ub5bc \uc124\uba85\ud574 \ubcf4\ub0b4\uc57c \ud55c\ub2e4.
- \ub9c8\ub298 srt\ud615\uc2dd\uc5d0 \uc900\uc218\ud558\uba70 \ud0c0\uc784\uc2a4\ud2b8\uacfc \uad00\ub828\ub41c \uc624\ub958\uac00 \uc544\ub2c8\uc5c8\uc73c\uba74 \ud55c\ub2e4. \ud2b9\ud788 9\ubd84\uc5d0\uc11c 1\uc2dc\uac04\uc73c\ub85c \uac74\ub108 \ub6f0\ub294 \ub4f1\uc758 \uc624\ub958\uac00 \uc544\ub2c8\uc5ec\uc57c \ud55c\ub2e4.

{source_lang} SRT:
{srt_chunk}
<end_of_turn>
<start_of_turn>model
"""
    
    return prompt

def translate_txt_chunks(chunks: List[str], source_lang: str = "\uc601\uc5b4", target_lang: str = "\ud55c\uad6d\uc5b4") -> List[str]:
    translated_chunks = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Translating TXT chunks", unit="chunk")):
        logger.info(f"Translating TXT chunk {i+1}/{len(chunks)}")
        
        try:
            prompt = create_translation_prompt(chunk, source_lang, target_lang)
            translated_chunk = call_llm_for_task(prompt, f"TXT Translation Chunk {i+1}")
            if any(marker in translated_chunk for marker in [
                TRANSLATION_FAILED_MARKER, PROCESSING_ERROR_MARKER, 
                CONTEXT_EXCEEDED_MARKER, REFINE_FAILED_MARKER
            ]):
                logger.error(f"Translation failed for chunk {i+1}: {translated_chunk[:200]}...")
                translated_chunks.append(f"[\uccad\ud06c {i+1} \ubc88\uc5ed \uc2e4\ud328]\n{chunk}")
            else:
                translated_chunks.append(translated_chunk)
                logger.info(f"Successfully translated chunk {i+1}")
            gpu_manager.cleanup()
            time.sleep(1.0)
        except Exception as e:
            logger.error(f"Error translating chunk {i+1}: {e}")
            translated_chunks.append(f"[\uccad\ud06c {i+1} \uc624\ub958: {str(e)}]\n{chunk}")
    return translated_chunks

def translate_srt_chunks(chunks: List[str], source_lang: str = "\uc601\uc5b4", target_lang: str = "\ud55c\uad6d\uc5b4") -> List[str]:
    translated_chunks = []
    for i, chunk in enumerate(tqdm(chunks, desc="Translating SRT chunks", unit="chunk")):
        logger.info(f"Translating SRT chunk {i+1}/{len(chunks)}")
        try:
            prompt = create_srt_translation_prompt(chunk, source_lang, target_lang)
            translated_chunk = call_llm_for_task(prompt, f"SRT Translation Chunk {i+1}")
            if any(marker in translated_chunk for marker in [
                TRANSLATION_FAILED_MARKER, PROCESSING_ERROR_MARKER, 
                CONTEXT_EXCEEDED_MARKER, SRT_GENERATION_FAILED_MARKER
            ]):
                logger.error(f"SRT translation failed for chunk {i+1}: {translated_chunk[:200]}...")
                translated_chunks.append(f"[\uccad\ud06c {i+1} \ubc88\uc5ed \uc2e4\ud328]\n{chunk}")
            else:
                translated_chunks.append(translated_chunk)
                logger.info(f"Successfully translated SRT chunk {i+1}")
            gpu_manager.cleanup()
            time.sleep(1.0)
        except Exception as e:
            logger.error(f"Error translating SRT chunk {i+1}: {e}")
            translated_chunks.append(f"[\uccad\ud06c {i+1} \uc624\ub958: {str(e)}]\n{chunk}")
    return translated_chunks

def translate_txt_file(file_path: Path, source_lang: str = "\uc601\uc5b4", target_lang: str = "\ud55c\uad6d\uc5b4", 
                       lines_per_chunk: int = 300) -> str:
    logger.info(f"Starting TXT file translation: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            logger.warning(f"Empty file: {file_path}")
            return ""
        chunks = split_txt_content(content, lines_per_chunk)
        if not chunks:
            logger.warning(f"No chunks created from file: {file_path}")
            return content
        translated_chunks = translate_txt_chunks(chunks, source_lang, target_lang)
        final_translation = '\n\n'.join(translated_chunks)
        output_path = file_path.parent / f"{file_path.stem}_translated_{target_lang}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_translation)
        logger.info(f"TXT translation completed: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error translating TXT file {file_path}: {e}")
        raise

def translate_srt_file(file_path: Path, source_lang: str = "\uc601\uc5b4", target_lang: str = "\ud55c\uad6d\uc5b4", 
                       entries_per_chunk: int = 300) -> str:
    logger.info(f"Starting SRT file translation: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            logger.warning(f"Empty file: {file_path}")
            return ""
        chunks = split_srt_content(content, entries_per_chunk)
        if not chunks:
            logger.warning(f"No chunks created from file: {file_path}")
            return content
        translated_chunks = translate_srt_chunks(chunks, source_lang, target_lang)
        final_translation = '\n\n'.join(translated_chunks)
        output_path = file_path.parent / f"{file_path.stem}_translated_{target_lang}.srt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_translation)
        logger.info(f"SRT translation completed: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error translating SRT file {file_path}: {e}")
        raise

def load_config_xlsx(config_path: str = "config.xlsx") -> pd.DataFrame:
    try:
        if not os.path.exists(config_path):
            logger.error(f"\uc124\uc815 \ud30c\uc77c\uc744 \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4: {config_path}")
            return pd.DataFrame()
        df = pd.read_excel(config_path)
        logger.info(f"\uc124\uc815 \ud30c\uc77c \ub85c\ub4dc \uc644\ub8cc: {config_path}")
        logger.info(f"\ucd1d {len(df)}\uac1c \ud56d\ubaa9 \ubc1c\uacac")
        return df
    except Exception as e:
        logger.error(f"\uc124\uc815 \ud30c\uc77c \uc77d\uae30 \uc624\ub958: {e}")
        return pd.DataFrame()

def get_files_from_config(config_df: pd.DataFrame) -> Tuple[List[Path], List[Path]]:
    txt_files = []
    srt_files = []
    if config_df.empty:
        logger.warning("\uc124\uc815 \ud30c\uc77c\uc774 \ube44\uc5b4\uc788\uc2b5\ub2c8\ub2e4.")
        return txt_files, srt_files
    if 'name' not in config_df.columns:
        logger.error(f"'name' \uc5f4\uc774 \uc5c6\uc2b5\ub2c8\ub2e4. \uc0ac\uc6a9 \uac00\ub2a5\ud55c \uc5f4\uc5b4: {list(config_df.columns)}")
        logger.info("config.xlsx \ud30c\uc77c\uc5d0 'name' \uc5f4\uc774 \uc788\ub294\uc9c0 \ud655\uc778\ud574\uc8fc\uc138\uc694.")
        return txt_files, srt_files
    logger.info(f"'name' \uc5f4\uc5d0\uc11c \ud30c\uc77c \uc815\ubcf4\ub97c \uc77d\uc2b5\ub2c8\ub2e4.")
    for idx, row in config_df.iterrows():
        name_value = row.get('name', '')
        if pd.isna(name_value) or not name_value:
            logger.warning(f"\ud589 {idx+1}: name \uac12\uc774 \ube44\uc5b4\uc788\uc2b5\ub2c8\ub2e4.")
            continue
        folder_name = name_value.strip()
        folder_path = Path(folder_name)
        file_base_name = f"{folder_name}_en"
        logger.debug(f"\ud589 {idx+1}: '{name_value}' \2192 \ud3f4\ub354: '{folder_name}', \ud30c\uc77c \uae30\ubc95\uba85: '{file_base_name}'")
        if not folder_path.exists() or not folder_path.is_dir():
            logger.warning(f"\ud3f4\ub354\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4: {folder_path}")
            continue
        srt_file_path = folder_path / f"{file_base_name}.srt"
        if srt_file_path.exists():
            srt_files.append(srt_file_path)
            logger.info(f"SRT \ud30c\uc77c \ubc1c\uacac: {srt_file_path}")
        else:
            logger.debug(f"SRT \ud30c\uc77c \uc5c6\uc74c: {srt_file_path}")
        txt_file_path = folder_path / f"{file_base_name}.txt"
        if txt_file_path.exists():
            txt_files.append(txt_file_path)
            logger.info(f"TXT \ud30c\uc77c \ubc1c\uacac: {txt_file_path}")
        else:
            logger.debug(f"TXT \ud30c\uc77c \uc5c6\uc74c: {txt_file_path}")
        if not srt_file_path.exists() and not txt_file_path.exists():
            logger.warning(f"\ud3f4\ub354 '{folder_path}'\uc5d0\uc11c '{file_base_name}.srt' \ub610\ub294 '{file_base_name}.txt' \ud30c\uc77c\uc744 \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.")
    logger.info(f"\uc124\uc815\uc5d0\uc11c \ubc1c\uacac\ub41c \ud30c\uc77c: TXT {len(txt_files)}\uac1c, SRT {len(srt_files)}\uac1c")
    return txt_files, srt_files

def should_translate_file(file_path: Path) -> bool:
    file_name = file_path.name.lower()
    if '_translated_' in file_name:
        return False
    skip_patterns = [
        '_backup',
        '_temp',
        '.log',
    ]
    for pattern in skip_patterns:
        if pattern in file_name:
            return False
    return True

def main_file_translation():
    logger.info("=== \ud30c\uc77c \ubc88\uc5ed \ud504\ub85c\uadf8\ub7a8 \uc2dc\uc791 ===")
    config_df = load_config_xlsx("config.xlsx")
    if config_df.empty:
        logger.error("config.xlsx \ud30c\uc77c\uc744 \uc77d\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.")
        logger.info("config.xlsx \ud30c\uc77c\uc774 \ud604\uc7ac \ub514\ub809\ud1a0\ub9ac\uc5d0 \uc788\ub294\uc9c0 \ud655\uc778\ud574\uc8fc\uc138\uc694.")
        logger.info("config.xlsx \ud30c\uc77c \ud615\uc2dd \uc608\uc2dc:")
        logger.info("| name                     | source_lang | target_lang |")
        logger.info("| Lesson 1 - Introduction | \uc601\uc5b4        | \ud55c\uad6d\uc5b4      |")
        logger.info("| Lesson 2 - Impermanence | \uc601\uc5b4        | \ud55c\uad6d\uc5b4      |")
        logger.info("")
        logger.info("\ud3f4\ub354 \uad6c\uc870 \uc608\uc2dc:")
        logger.info("Lesson 1 - Introduction/")
        logger.info("  \u251c\u2014 Lesson 1 - Introduction_en.srt")
        logger.info("  \u2514\u2014 Lesson 1 - Introduction_en.txt")
        logger.info("Lesson 2 - Impermanence/")
        logger.info("  \u251c\u2014 Lesson 2 - Impermanence_en.srt")
        logger.info("  \u2514\u2014 Lesson 2 - Impermanence_en.txt")
        return
    all_txt_files, all_srt_files = get_files_from_config(config_df)
    txt_files = [f for f in all_txt_files if should_translate_file(f)]
    srt_files = [f for f in all_srt_files if should_translate_file(f)]
    excluded_txt = [f for f in all_txt_files if not should_translate_file(f)]
    excluded_srt = [f for f in all_srt_files if not should_translate_file(f)]
    current_dir = Path('.').absolute()
    logger.info(f"\ud604\uc7ac \ub514\ub809\ud1a0\ub9ac: {current_dir}")
    if excluded_txt or excluded_srt:
        logger.info(f"\uc81c\uc81c\ub41c \ud30c\uc77c\ub4e4:")
        for file in excluded_txt + excluded_srt:
            logger.info(f"  - {file.name} (\ubc88\uc5ed \ubd88\ud544\uc694)")
    if txt_files:
        logger.info(f"\ubc88\uc5ed\ud560 TXT \ud30c\uc77c: {len(txt_files)}\uac1c")
        for file in txt_files:
            logger.info(f"  - {file}")
    if srt_files:
        logger.info(f"\ubc88\uc5ed\ud560 SRT \ud30c\uc77c: {len(srt_files)}\uac1c")
        for file in srt_files:
            logger.info(f"  - {file}")
    if not txt_files and not srt_files:
        logger.warning("\ubc88\uc5ed\ud560 TXT \ub610\ub294 SRT \ud30c\uc77c\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.")
        logger.info("\ub2e4\uc74c \uc0c1\ud0dc\ub97c \ud655\uc778\ud574\uc8fc\uc138\uc694:")
        logger.info("1. config.xlsx\uc758 'name' \uc5f4\uc5d0 \uc62c\ubc14\ub978 \ud3f4\ub354\uba85\uc774 \uc788\ub294\uc9c0")
        logger.info("2. \ud574\ub2f9 \ud3f4\ub354\uac00 \uc874\uc7ac\ud558\ub294\uc9c0 (\uc608: 'Lesson 2 - Impermanence')")
        logger.info("3. \ud3f4\ub354 \uc548\uc5d0 '_en'\uc774 \ub2f4\uacbc\uc74c .srt \ub610\ub294 .txt \ud30c\uc77c\uc774 \uc788\ub294\uc9c0")
        logger.info("   (\uc608: 'Lesson 2 - Impermanence_en.srt')")
        return
    source_lang = "\uc601\uc5b4"
    target_lang = "\ud55c\uad6d\uc5b4"
    if 'source_lang' in config_df.columns:
        source_lang = config_df['source_lang'].iloc[0] if not config_df.empty else "\uc601\uc5b4"
    if 'target_lang' in config_df.columns:
        target_lang = config_df['target_lang'].iloc[0] if not config_df.empty else "\ud55c\uad6d\uc5b4"
    logger.info(f"\ubc88\uc5ed \uc124\uc815: {source_lang} \u2192 {target_lang}")
    translated_files = []
    for txt_file in txt_files:
        try:
            logger.info(f"Processing TXT file: {txt_file}")
            output_file = translate_txt_file(txt_file, source_lang, target_lang)
            translated_files.append(output_file)
        except Exception as e:
            logger.error(f"Failed to translate TXT file {txt_file}: {e}")
    for srt_file in srt_files:
        try:
            logger.info(f"Processing SRT file: {srt_file}")
            output_file = translate_srt_file(srt_file, source_lang, target_lang)
            translated_files.append(output_file)
        except Exception as e:
            logger.error(f"Failed to translate SRT file {srt_file}: {e}")
    logger.info("=== \ubc88\uc5ed \uc644\ub8cc ===")
    logger.info(f"\ucd1d {len(translated_files)}\uac1c \ud30c\uc77c\uc774 \ubc88\uc5ed\ub418\uc5c8\uc2b5\ub2c8\ub2e4:")
    for file in translated_files:
        logger.info(f"  - {file}")

def generate_text_for_chunk(prompt: str, max_new_tokens: int, task_description: str = "Inference") -> str:
    input_token_count = text_processor.count_tokens(prompt)
    logger.debug(f"Task '{task_description}': Input Tokens: {input_token_count}, Prompt len: {len(prompt)} chars, Max new: {max_new_tokens}")
    max_new_tokens = max(64, max_new_tokens)
    if input_token_count + max_new_tokens >= N_CTX:
        original_max_new = max_new_tokens
        max_new_tokens = N_CTX - input_token_count - 100
        logger.warning(f"Ctx risk: Input({input_token_count})+Req({original_max_new})>=Ctx({N_CTX}). Adjusting max_new to {max_new_tokens}.")
        if max_new_tokens <= 0:
            logger.error(f"{task_description} failed: Context window exceeded even before generation.")
            return f"[\uc624\ub958] {task_description} \uc2e4\ud328: \ucf58\ud150\uc2a4\ud2b8 \ucd5c\uac15 \uacb9\ubd80"
    try:
        output = model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            stop=["<|end_of_text|>", "</s>", "\n\n---", "##"],
            echo=False
        )
        generated_text = output['choices'][0]['text'].strip()
        output_token_count = text_processor.count_tokens(generated_text)
        logger.debug(f"{task_description} OK. Output tokens: {output_token_count}.")
        cleaned_text = generated_text
        instruction_patterns = [
            "(\ubc88\uc5ed \uacb0\uacfc\ub9cc \uc81c\uacf5\ud574\uc8fc\uc138\uc694)",
            "\ubc88\uc5ed \uacb0\uacfc:",
            "\ubc88\uc5ed:",
            "Translation:",
            "Result:"
        ]
        for pattern in instruction_patterns:
            if cleaned_text.endswith(pattern):
                cleaned_text = cleaned_text[:-len(pattern)].rstrip()
            if cleaned_text.startswith(pattern):
                cleaned_text = cleaned_text[len(pattern):].lstrip()
        gpu_manager.cleanup()
        return cleaned_text
    except Exception as e:
        logger.error(f"Error during {task_description}: {e}", exc_info=True)
        gpu_manager.cleanup()
        return f"[\uc624\ub958] {task_description} \uc2e4\ud328: {str(e)}"

def generate_text(prompt: str, task_description: str = "Inference") -> str:
    input_tokens = text_processor.count_tokens(prompt)
    is_korean_input = text_processor.contains_korean(prompt)
    estimated_output_tokens = text_processor.estimate_output_tokens(prompt, not is_korean_input, 1.8 if not is_korean_input else 1.0)
    total_estimated_tokens = input_tokens + estimated_output_tokens
    need_chunking = (input_tokens > CHUNK_SIZE_TOKENS) or FORCE_CHUNK
    logger.info(f"{task_description}: Input tokens: {input_tokens}, Est. output: {estimated_output_tokens}")
    single_call_max_new_tokens = min(BASE_MAX_TOKENS, estimated_output_tokens)
    can_process_single = (input_tokens + single_call_max_new_tokens < N_CTX)
    if not need_chunking and can_process_single:
        logger.info(f"Processing one go (max new: {single_call_max_new_tokens})")
        return generate_text_for_chunk(prompt, single_call_max_new_tokens, task_description)
    else:
        if not can_process_single and not need_chunking:
            logger.warning(f"Short text but est. total({total_estimated_tokens})>Ctx({N_CTX}). Forcing chunking.")
        logger.info(f"Chunking required (Input>{CHUNK_SIZE_TOKENS} or forced/context).")
        optimal_chunk_size, optimal_overlap = text_processor.get_optimal_chunk_size(prompt, N_CTX, CHUNK_OVERLAP_TOKENS)
        chunks = text_processor.chunk_text(prompt, optimal_chunk_size, optimal_overlap)
        logger.info(f"Split into {len(chunks)} chunks.")
        translated_chunks = []
        for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {task_description} Chunks", unit="chunk")):
            chunk_input_tokens = text_processor.count_tokens(chunk)
            logger.info(f"Chunk {i+1}/{len(chunks)} ({chunk_input_tokens} tokens)")
            chunk_max_new_tokens = min(N_CTX - chunk_input_tokens - 50, BASE_MAX_TOKENS)
            if chunk_max_new_tokens <= 0:
                logger.error(f"Chunk {i+1} too large ({chunk_input_tokens}) for Ctx({N_CTX}). Skip.")
                translated_chunks.append(f"[\uc624\ub958] \uccad\ud06c{i+1} \ucf58\ud150\uc2a4\ud2b8 \ucd5c\uac15 \uacb9\ubd80")
                continue
            logger.debug(f"Chunk {i+1}: Requesting max {chunk_max_new_tokens} new tokens.")
            chunk_result = generate_text_for_chunk(chunk, chunk_max_new_tokens, f"{task_description} Chunk {i+1}/{len(chunks)}")
            translated_chunks.append(chunk_result)
            gpu_manager.cleanup()
            time.sleep(0.5)
        logger.info(f"Merging {len(chunks)} chunks...")
        merged_result = text_processor.merge_translations(chunks, translated_chunks, optimal_overlap)
        logger.info(f"Merged len: {len(merged_result)} chars")
        return merged_result

RETRY_WAIT = wait_exponential(multiplier=1, min=4, max=60)
RETRY_EXCEPTIONS = (RuntimeError, ValueError, OSError)

def log_retry_failure(retry_state):
    logger.error(f"Model call failed after {retry_state.attempt_number} attempts. Last exception: {retry_state.outcome.exception()}")

@retry(
    stop=stop_after_attempt(3),
    wait=RETRY_WAIT,
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    retry_error_callback=log_retry_failure
)
def gem_call(prompt: str, task_description: str = "Inference") -> str:
    return generate_text(prompt, task_description)

TRANSLATION_FAILED_MARKER = "[\ubc88\uc5ed \uc2e4\ud328]"
REFINE_FAILED_MARKER = "[\uc719\ubb38 \uc2e4\ud328]"
PROCESSING_ERROR_MARKER = "[\ucc98\ub9ac \uc624\ub958]"
SRT_GENERATION_FAILED_MARKER = "[SRT \uc0dd\uc131 \uc2e4\ud328]"
CONTEXT_EXCEEDED_MARKER = "[\ucf58\ud150\uc2a4\ud2b8 \ucd5c\uac15 \uacb9\ubd80 \uc624\ub958]"

def call_llm_for_task(prompt: str, task_description: str) -> str:
    try:
        result = gem_call(prompt, task_description=task_description)
        if any(marker in result for marker in [TRANSLATION_FAILED_MARKER, REFINE_FAILED_MARKER, PROCESSING_ERROR_MARKER, SRT_GENERATION_FAILED_MARKER, CONTEXT_EXCEEDED_MARKER]):
            logger.warning(f"{task_description} result has marker: {result[:200]}...")
            return result
        logger.info(f"{task_description} task successful.")
        return result
    except (RetryError, RuntimeError, Exception) as e:
        logger.error(f"{task_description} failed: {e}", exc_info=False)
        original_error = e.__cause__ if hasattr(e, '__cause__') and e.__cause__ else e
        err_str = str(original_error).lower()
        failure_prefix = f"[{task_description} \uc2e4\ud328]"
        if "context window" in err_str or "exceeds context" in err_str or "n_ctx" in err_str:
            return f"{CONTEXT_EXCEEDED_MARKER} ({original_error})"
        elif isinstance(e, RetryError):
            last_exception = e.last_attempt.exception()
            err_str_last = str(last_exception).lower()
            if "context window" in err_str_last or "exceeds context" in err_str_last or "n_ctx" in err_str_last:
                return f"{CONTEXT_EXCEEDED_MARKER} (Retry limit: {last_exception})"
            else:
                return f"{PROCESSING_ERROR_MARKER} (Retry limit: {last_exception})"
        elif "ggml" in err_str or "llama_" in err_str:
            return f"{PROCESSING_ERROR_MARKER} (llama.cpp error: {original_error})"
        else:
            return f"{PROCESSING_ERROR_MARKER} (\uc624\ub958: {original_error})"

if __name__ == "__main__":
    main_file_translation()
