const API_BASE = window.API_BASE || "http://localhost:8000";

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadResult = document.getElementById("uploadResult");
const processForm = document.getElementById("processForm");
const processFilename = document.getElementById("processFilename");
const processResult = document.getElementById("processResult");
const output = document.getElementById("output");

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    uploadResult.innerHTML = '<div class="result-box">Uploading...</div>';

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            uploadResult.innerHTML = `<div class="result-box error">${data.detail || data.message}</div>`;
            return;
        }

        uploadResult.innerHTML = `
            <div class="result-box success">
                <strong>Uploaded:</strong> ${data.data.original_filename}<br>
                <strong>Stored as:</strong> ${data.data.stored_filename}<br>
                <strong>Scan:</strong> ${data.data.scan.safe ? "Safe" : "Issues found"}
                <pre>${JSON.stringify(data.data.scan, null, 2)}</pre>
            </div>`;
        processFilename.value = data.data.stored_filename;
    } catch (err) {
        uploadResult.innerHTML = `<div class="result-box error">Connection failed</div>`;
    }
});

processForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const filename = processFilename.value;
    if (!filename) {
        processResult.innerHTML = '<div class="result-box error">Upload a file first</div>';
        return;
    }

    const checkboxes = document.querySelectorAll(".operations input[type='checkbox']:checked");
    const operations = Array.from(checkboxes).map((cb) => cb.value);

    if (operations.length === 0) {
        processResult.innerHTML = '<div class="result-box error">Select at least one operation</div>';
        return;
    }

    processResult.innerHTML = '<div class="result-box">Processing...</div>';

    try {
        const res = await fetch(`${API_BASE}/process?filename=${encodeURIComponent(filename)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ operations }),
        });
        const data = await res.json();

        if (!res.ok) {
            processResult.innerHTML = `<div class="result-box error">${data.detail || data.message}</div>`;
            return;
        }

        processResult.innerHTML = `<div class="result-box success">${data.message}</div>`;
        showOutput(data.data.processed_filename);
    } catch (err) {
        processResult.innerHTML = `<div class="result-box error">Connection failed</div>`;
    }
});

function showOutput(filename) {
    const downloadUrl = `${API_BASE}/download/${encodeURIComponent(filename)}`;
    output.innerHTML = `
        <img class="preview" src="${downloadUrl}" alt="Processed image"
             onerror="this.style.display='none'">
        <a class="download-link" href="${downloadUrl}" download="${filename}">
            Download Processed Image
        </a>`;
}
