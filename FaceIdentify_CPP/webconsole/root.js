// Connection status
const connectionStatus = document.getElementById('connectionStatus');

// Controls
const showRawBtn = document.getElementById('show_raw_btn');
const showProcessedBtn = document.getElementById('show_processed_btn');
const recordBtn = document.getElementById('record_btn');
const recordText = document.getElementById('record_text');
const cameraIcon = document.getElementById('camera_icon');

// Containers
const processedContainer = document.getElementById('processed_container');
const rawContainer = document.querySelector('.video-container:not(#processed_container)'); // Get the raw camera container

// Button handlers
showRawBtn.onclick = function () {
    const isActive = this.classList.contains('active');

    showRawBtn.classList.remove('active');
    showProcessedBtn.classList.remove('active');
    processedContainer.style.display = 'none';
    rawContainer.style.display = 'none'; // Hide raw container initially

    if (!isActive) {
        this.classList.add('active');
        rawContainer.style.display = 'block'; // Show only raw camera
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "process", value: false }));
        }
    }
};

showProcessedBtn.onclick = function () {
    const isActive = this.classList.contains('active');

    showRawBtn.classList.remove('active');
    showProcessedBtn.classList.remove('active');
    processedContainer.style.display = 'none';
    rawContainer.style.display = 'none'; // Hide raw container

    if (!isActive) {
        this.classList.add('active');
        processedContainer.style.display = 'block'; // Show only processed camera

        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "process", value: true }));
        }
    } else {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "process", value: false }));
        }
    }
};

// Recording functionality
let isRecording = false;
recordBtn.onclick = function () {
    isRecording = !isRecording;

    if (isRecording) {
        recordBtn.classList.add('recording');
        recordText.textContent = "Đang ghi hình";
        cameraIcon.innerHTML = `
                    <svg class="icon" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="8" fill="currentColor"/>
                    </svg>
                `;
    } else {
        recordBtn.classList.remove('recording');
        recordText.textContent = "Bắt đầu ghi hình";
        cameraIcon.innerHTML = `
                    <svg class="icon" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="3"/>
                        <path d="M12 1L9 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-4l-3-3z"/>
                    </svg>
                `;
    }

    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "recording", value: isRecording }));
    }
};

// WebSocket connection
const ws = new WebSocket("ws://localhost:9002");
ws.binaryType = "arraybuffer";

const raw_canvas = document.getElementById("raw_canvas");
const ctx_raw_canvas = raw_canvas.getContext("2d");
const processed_canvas = document.getElementById("processed_canvas");
const ctx_processed_canvas = processed_canvas.getContext("2d");

ws.onopen = () => {
    connectionStatus.textContent = "Đã kết nối";
    connectionStatus.className = "connection-status connected";
};

ws.onclose = () => {
    connectionStatus.textContent = "Mất kết nối";
    connectionStatus.className = "connection-status disconnected";
};

ws.onerror = () => {
    connectionStatus.textContent = "Lỗi kết nối";
    connectionStatus.className = "connection-status disconnected";
};

ws.onmessage = (event) => {
    if (typeof event.data === "string") {
        try {
            const msg = JSON.parse(event.data);

            if (msg.type === "raw" || msg.type === "processed") {
                const img = new Image();
                img.onload = () => {
                    if (msg.type === "raw") {
                        raw_canvas.width = img.width;
                        raw_canvas.height = img.height;
                        ctx_raw_canvas.drawImage(img, 0, 0);
                    } else if (msg.type === "processed") {
                        processed_canvas.width = img.width;
                        processed_canvas.height = img.height;
                        ctx_processed_canvas.drawImage(img, 0, 0);
                    }
                };
                img.src = "data:image/jpeg;base64," + msg.image;
            }

            if (msg.type === "identify" && msg.info) {
                const table = document.getElementById('identify_table').getElementsByTagName('tbody')[0];
                table.innerHTML = "";
                const row = table.insertRow();
                row.insertCell(0).textContent = msg.info.id;
                row.insertCell(1).textContent = msg.info.name;
                row.insertCell(2).textContent = msg.info.old;
                row.insertCell(3).textContent = new Date().toLocaleString('vi-VN');
            }
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    }
};

// Form submission
document.getElementById('add_identify_form').onsubmit = function () {
    const name = document.getElementById('input_name').value.trim();
    const old = document.getElementById('input_old').value.trim();

    const nameRegex = /^[A-Za-zÀ-ỹà-ỹ\s]{2,}$/u;
    if (!nameRegex.test(name)) {
        alert("Vui lòng nhập tên hợp lệ (chỉ chữ cái, tối thiểu 2 ký tự, không số hoặc ký tự đặc biệt)!");
        document.getElementById('input_name').focus();
        return false;
    }
    if (name.length > 50) {
        alert("Tên không được dài quá 50 ký tự!");
        document.getElementById('input_name').focus();
        return false;
    }

    const oldNum = Number(old);
    if (!Number.isInteger(oldNum) || oldNum < 1 || oldNum > 120) {
        alert("Vui lòng nhập tuổi hợp lệ (1-120)!");
        document.getElementById('input_old').focus();
        return false;
    }

    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: "add_identify",
            info: {
                name: name,
                old: oldNum
            }
        }));
    }

    document.getElementById('input_name').value = '';
    document.getElementById('input_old').value = '';
    return false;
};

// Initialize - show raw camera by default
showRawBtn.classList.add('active');
rawContainer.style.display = 'block';
processedContainer.style.display = 'none';

// Shutdown functionality
const shutdownBtn = document.getElementById('shutdown_btn');
shutdownBtn.onclick = function () {
    if (confirm("Bạn có chắc chắn muốn tắt hệ thống?")) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "shutdown", value: true }));
            connectionStatus.textContent = "Đang tắt hệ thống...";
            connectionStatus.className = "connection-status disconnected";
        }
    }
};

// Guide modal functionality
const guideModal = document.getElementById('guideModal');
const guideBtn = document.getElementById('guide_btn');
const closeGuideBtn = document.getElementById('closeGuideBtn');
const printGuideBtn = document.getElementById('printGuideBtn');
const gotItBtn = document.getElementById('gotItBtn');

// Show guide button functionality
guideBtn.onclick = function () {
    guideModal.style.display = 'block';
};

// Open guide modal on first visit
let isFirstVisit = localStorage.getItem('isFirstVisit') === null;
if (isFirstVisit) {
    guideModal.style.display = 'block';
    localStorage.setItem('isFirstVisit', 'true');
}

closeGuideBtn.onclick = function () {
    guideModal.style.display = 'none';
};

printGuideBtn.onclick = function () {
    window.print();
};

gotItBtn.onclick = function () {
    guideModal.style.display = 'none';
};

// Close modal when clicking outside
guideModal.onclick = function (event) {
    if (event.target === guideModal) {
        guideModal.style.display = 'none';
    }
};
