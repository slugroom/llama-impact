const sendData = blob => {
    let recordedAudio = document.getElementById("recordedAudio");
    recordedAudio.src = URL.createObjectURL(blob);
    recordedAudio.controls=true;
    recordedAudio.autoplay=true;
};

const toggleRecording = () => {
    audioChunks = [];
    if (recording) rec.stop();
    else rec.start();
    recording = !recording;
};

var rec, recording = false, audioChunks = [];
const audioHandler = stream => {
    rec = new MediaRecorder(stream);
    rec.addEventListener("dataavailable", e => {
        audioChunks.push(e.data);
        if (rec.state == "inactive") {
            let blob = new Blob(audioChunks, { type: "audio/mpeg3" });
            sendData(blob);
        }
    });
    toggleRecording();
};

document.getElementById("record-button").addEventListener("click", () => {
    if (!rec) navigator.mediaDevices.getUserMedia({ audio: true }).then(audioHandler);
    else toggleRecording();
});