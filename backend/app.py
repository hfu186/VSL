import os
import logging
import re
from typing import Dict, Optional
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import pandas as pd
import librosa
import soundfile as sf
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import traceback

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sigml_app")

# =========================
# PATH SETUP (QUAN TRỌNG)
# =========================
HERE = Path(__file__).resolve().parent        # backend/
PROJECT_ROOT = HERE.parent                   # Demo/

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL = "vinai/PhoWhisper-medium"

BARTPHO_DIR = HERE / "models" / "final_model"
VOCAB_CSV = HERE / "data" / "vocab_sigml.csv"
SIGML_PATH = HERE / "sigml" / "output.sigml"

VNCORENLP_JAR = HERE / "models" / "vncorenlp_model" / "VnCoreNLP-1.2.jar"

SAMPLE_RATE = 16000

# =========================
# GLOBAL OBJECTS
# =========================
whisper_processor = None
whisper_model = None
bartpho_tokenizer = None
bartpho_model = None
vncorenlp_client = None
VOCAB: Dict[str, str] = {}

# =========================
# FASTAPI LIFESPAN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_processor, whisper_model
    global bartpho_tokenizer, bartpho_model
    global vncorenlp_client, VOCAB

    # -------- STARTUP --------
    logger.info("Server starting...")
    logger.info(f"CWD={Path.cwd()}")
    logger.info(f"VOCAB path={VOCAB_CSV} exists={VOCAB_CSV.exists()}")

    # Load vocab
    if VOCAB_CSV.exists():
        df = pd.read_csv(VOCAB_CSV, encoding="utf-8")
        for _, row in df.iterrows():
            token = str(row["token"]).strip().lower()
            sigml = str(row["sigml"]).strip()
            if token:
                VOCAB[token] = sigml
        logger.info(f"Loaded vocab entries: {len(VOCAB)}")
    else:
        logger.warning("VOCAB CSV not found – vocab empty")

    # Load Whisper
    whisper_processor = WhisperProcessor.from_pretrained(ASR_MODEL)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL)
    whisper_model.to(DEVICE).eval()
    logger.info("Whisper loaded")

    # Load BARTpho
    bartpho_tokenizer = AutoTokenizer.from_pretrained(BARTPHO_DIR)
    bartpho_model = AutoModelForSeq2SeqLM.from_pretrained(BARTPHO_DIR)
    bartpho_model.to(DEVICE).eval()
    logger.info("BARTpho loaded")

    # Load VnCoreNLP (official)
    try:
        from vncorenlp import VnCoreNLP
        vncorenlp_client = VnCoreNLP(
            str(VNCORENLP_JAR),
            annotators="wseg",
            max_heap_size="-Xmx2g"
        )
        logger.info("VnCoreNLP loaded")
    except Exception as e:
        logger.warning(f"VnCoreNLP disabled: {e}")
        vncorenlp_client = None

    yield

    # -------- SHUTDOWN --------
    logger.info("Server shutting down")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Speech2Sign", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# TOKENIZATION
# =========================
_word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_text(text: str):
    if vncorenlp_client:
        sents = vncorenlp_client.tokenize(text)
        return [tok.replace(" ", "_").lower() for sent in sents for tok in sent]
    return [t.lower() for t in _word_re.findall(text) if t.strip()]

# =========================
# SIGML GENERATION
# =========================
def generate_sigml_from_gloss(gloss: str) -> str:
    tokens = tokenize_text(gloss)
    fragments = []

    for tok in tokens:
        if tok in VOCAB:
            fragments.append(VOCAB[tok])
        else:
            for ch in tok:
                if ch in VOCAB:
                    fragments.append(VOCAB[ch])

    return f"<sigml>{''.join(fragments)}</sigml>"

# =========================
# ASR
# =========================
def transcribe_audio(path: str) -> str:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    inputs = whisper_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = whisper_model.generate(inputs.input_features.to(DEVICE))
    return whisper_processor.batch_decode(out, skip_special_tokens=True)[0]

# =========================
# BARTPHO
# =========================
def bartpho_translate(text: str) -> str:
    inputs = bartpho_tokenizer(
        text,
        return_tensors="pt"
    )

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = bartpho_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=256
        )

    return bartpho_tokenizer.decode(
        out[0],
        skip_special_tokens=True
    )

# =========================
# API
# =========================
@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    try:
        # 1. Save uploaded file
        tmp = HERE / file.filename
        with open(tmp, "wb") as f:
            f.write(await file.read())

        # 2. Process pipeline
        text = transcribe_audio(str(tmp))
        print("Transcribed text:", text)
        gloss = bartpho_translate(text)
        print("Generated gloss:", gloss)
        sigml = generate_sigml_from_gloss(gloss)
        print("Generated SiGML:", sigml)

        # 3. Save SiGML
        SIGML_PATH.parent.mkdir(parents=True, exist_ok=True)
        SIGML_PATH.write_text(sigml, encoding="utf-8")

        # 4. Success response
        return {
            "status": "success",
            "text": text,
            "gloss": gloss,
            "sigml": sigml,
        }

    except Exception as e:
        # log chi tiết để debug server
        print("Error in /process_audio")
        traceback.print_exc()

        return {
            "status": "fail",
            "message": str(e),
        }
