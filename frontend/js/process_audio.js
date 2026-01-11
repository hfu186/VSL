
/**
 * @param {number} multiplier 
 */

function playSigml(sigml) {
  if (!sigml || !window.CWASA) return;

  if (CWASA.stop) {
    CWASA.stop();
  }
  CWASA.playSiGMLText(sigml);
  setTimeout(() => {
  }, 120); 
}


window.repeatAvatar = function () {
  if (!window.lastSigmlText) {
    alert("Chưa có nội dung để phát lại.");
    return;
  }

  playSigml(window.lastSigmlText);
};


async function processText() {
  const textInput = document.getElementById("textInput");
  const text = textInput.value.trim();

  if (!text) {
    alert("Vui lòng nhập văn bản");
    return;
  }

  updateTranscription("Đang xử lý...");
  updateGloss("Processing");

  try {
    const response = await fetch("http://127.0.0.1:8000/process_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data.status !== "success") {
      throw new Error(data.message || "Backend error");
    }

    updateTranscription(text);
    updateGloss(data.gloss || "N/A");

    if (!data.sigml) {
      throw new Error("Không có dữ liệu SiGML");
    }

    window.lastSigmlText = data.sigml;
    playSigml(data.sigml);

  } catch (err) {
    console.error(err);
    alert("Lỗi xử lý văn bản: " + err.message);
  }
}
async function startRecording() {
  try {
    const recordBtn = document.getElementById("recordBtn");
    const recordingUI = document.getElementById("recordingUI");

    recordingUI.classList.remove("hidden");
    recordBtn.disabled = true;

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
    });

    const input = audioContext.createMediaStreamSource(stream);

    recorder = new Recorder(input, {
      numChannels: 1,
    });

    recorder.record();
    isRecording = true;

    console.log("Recording started");
  } catch (error) {
    console.error("Error starting recording:", error);
    alert("Unable to access the microphone. Please allow microphone access.");
    resetRecordingUI();
  }
}

function resetRecordingUI() {
  const recordBtn = document.getElementById("recordBtn");
  const recordingUI = document.getElementById("recordingUI");

  if (recordBtn) recordBtn.disabled = false;
  if (recordingUI) recordingUI.classList.add("hidden");
}

async function stopRecording() {
  if (!isRecording || !recorder) {
    console.warn("No active recording to stop");
    return;
  }

  try {
    updateTranscription("Processing audio...");

    recorder.stop();
    isRecording = false;

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }

    recorder.exportWAV(async blob => {
      const file = new File([blob], "recorded.wav", {
        type: "audio/wav",
      });

      console.log("Recorded audio size:", blob.size, "bytes");
      await sendToBackend(file);
    });

    resetRecordingUI();
  } catch (error) {
    console.error("Error stopping recording:", error);
    alert("Error stopping recording: " + error.message);
    resetRecordingUI();
  }
}

async function uploadAudio() {
  const input = document.getElementById("audioFile");

  if (!input.files || !input.files.length) {
    alert("Please select an audio file");
    return;
  }

  const file = input.files[0];
  console.log("Uploading file:", file.name, file.type, file.size, "bytes");

  updateTranscription("Processing audio file...");
  await sendToBackend(file);
}
async function sendToBackend(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/process_audio", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (data.status !== "success") {
      throw new Error(data.message || "Audio processing failed");
    }

    updateTranscription(data.transcription || "Không nhận diện được");
    updateGloss(data.gloss || "N/A");

    if (!data.sigml) {
      throw new Error("Không có dữ liệu SiGML");
    }

    window.lastSigmlText = data.sigml;
    playSigml(data.sigml);

  } catch (err) {
    console.error(err);
    alert("Lỗi xử lý audio: " + err.message);
  }
}

function updateTranscription(text) {
  const el = document.getElementById("transcriptionOutput");
  if (el) el.innerText = text;
}

function updateGloss(text) {
  const el = document.getElementById("glossOutput");
  if (el) el.innerText = "Gloss: " + text;
}

document.addEventListener("DOMContentLoaded", () => {
  console.log("CWASA Player loaded");

  if (typeof CWASA === "undefined") {
    console.warn("CWASA not loaded");
  }
});
