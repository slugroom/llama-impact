
const MAX_TASKNAME_LENGTH = 61;

const uploadButton = document.getElementById("upload-button");
const inputElement = document.getElementById("drop-input");

var tasks = [];

const createTask = taskName => {
    if (taskName.length > MAX_TASKNAME_LENGTH) taskName = taskName.slice(0, MAX_TASKNAME_LENGTH) + "...";
    if (tasks.filter(x => x).length === 0) document.getElementById("task-message").style.display = "none";
    const taskElement = document.createElement("div");
    taskElement.classList.add("task");
    taskElement.innerHTML = `
        <div class='task-name' onclick='openTask(${tasks.length});'>Fetching...</div>
        <button class='task-button edit-button' onclick='editTask(${tasks.length});' type='button'></button>
        <button class='task-button delete-button' onclick='deleteTask(${tasks.length}, this);' type='button'></button>
    `;
    document.getElementById("tasks-wrapper").appendChild(taskElement);
    tasks.push([
        taskElement,
        "No text available yet. Processing is in progress...",
        "No text available yet. Processing is in progress...",
        false,
        taskName
    ]);
};

const openTask = index => {
    const taskElements = document.getElementsByClassName("task");
    for (let i = 0; i < taskElements.length; i++) {
        taskElements[i].style.opacity = "";
    }
    tasks[index][0].style.opacity = "1";

    document.getElementById("original-text").innerText = tasks[index][1];
    document.getElementById("corrected-text").innerText = tasks[index][2];
};

const editTask = index => {
    const newName = prompt("Enter a new name for the task:");
    if (!newName) return;
    tasks[index][0].getElementsByClassName("task-name")[0].innerText = newName;
};

const deleteTask = (index, elm) => {
    if (tasks[index][3]) {
        elm.parentElement.remove();
        tasks[index] = undefined;
        if (tasks.filter(x => x).length === 0) document.getElementById("task-message").style.display = "";
    } else alert("Please wait for the task to finish.");
};

const sendData = async (blob, filename) => {
    const index = JSON.parse(JSON.stringify(tasks.length));
    const formData = new FormData();
    formData.append("audio_data", blob, "audio.mp3");

    const apiUrl = "/data-send";
    createTask(filename);
    const response = await fetch(apiUrl, {
        method: "POST",
        cache: "no-cache",
        body: formData
    });
    if (response.ok) {
        const { task_id } = await response.json();
        
        checkTaskStatus(task_id, index);
    } else {
        tasks[index][1] = "Error sending data.";
        tasks[index][2] = "Error sending data.";
        tasks[index][3] = true;
        tasks[index][4] = "Failed!";
        tasks[index][0].getElementsByClassName("task-name")[0].innerText = tasks[index][4];
    }
};

const checkTaskStatus = async (task_id, index) => {
    const statusUrl = `/task-status/${task_id}`;
    
    const poll = async () => {
        const response = await fetch(statusUrl);
        if (response.ok) {
            const taskInfo = await response.json();
            
            if (taskInfo.status === "completed") {
                tasks[index][1] = taskInfo.result.original;
                tasks[index][2] = taskInfo.result.corrected;
                tasks[index][3] = true;
                tasks[index][0].getElementsByClassName("task-name")[0].innerText = tasks[index][4];
            } else if (taskInfo.status === "failed") {
                tasks[index][1] = "Error processing data.";
                tasks[index][2] = "Error processing data.";
                tasks[index][3] = true;
                tasks[index][4] = "Failed!";
                tasks[index][0].getElementsByClassName("task-name")[0].innerText = tasks[index][4];
                console.error(taskInfo.error);
            } else {
                setTimeout(poll, 3000);
            }
        } else {
            tasks[index][1] = "Error retrieving status.";
            tasks[index][2] = "Error retrieving status.";
            tasks[index][3] = true;
            tasks[index][4] = "Failed!";
            tasks[index][0].getElementsByClassName("task-name")[0].innerText = tasks[index][4];
        }
    };
    
    poll();
};


const toggleRecording = () => {
    audioChunks = [];
    if (recording) rec.stop();
    else rec.start();
    document.getElementById("record-button").style.aspectRatio = recording ? "1 / 1" : "";
    recording = !recording;
};

var rec, recording = false, audioChunks = [];
const audioHandler = stream => {
    toggleRecording();
    rec = new MediaRecorder(stream);
    rec.addEventListener("dataavailable", e => {
        audioChunks.push(e.data);
        if (rec.state == "inactive") {
            let blob = new Blob(audioChunks, { type: "audio/mpeg3" });
            sendData(blob, "Recording_" + Date.now());
        }
    });
};

const queueUpload = () => {
    const files = inputElement.files;
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if(file.type === "audio/mpeg") {
            sendData(file, file.name);
        }
    }
};

document.getElementById("record-button").addEventListener("click", () => {
    if (!rec) navigator.mediaDevices.getUserMedia({ audio: true }).then(audioHandler);
    else toggleRecording();
});

uploadButton.addEventListener("click", () => inputElement.click());

inputElement.addEventListener("change", () => {
    if (inputElement.files.length) queueUpload();
});

uploadButton.addEventListener("dragover", e => {
    e.preventDefault();
    uploadButton.style.aspectRatio = "1 / 1";
});

["dragleave", "dragend"].forEach((type) => {
    uploadButton.addEventListener(type, () => {
        uploadButton.style.aspectRatio = "";
    });
});

uploadButton.addEventListener("drop", e => {
    e.preventDefault();

    if (e.dataTransfer.files.length) {
        inputElement.files = e.dataTransfer.files;
        queueUpload();
    }
    uploadButton.style.aspectRatio = "";
});
