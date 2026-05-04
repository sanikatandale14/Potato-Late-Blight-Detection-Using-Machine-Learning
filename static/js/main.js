const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const changeImageBtn = document.getElementById('changeImageBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

let selectedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

changeImageBtn.addEventListener('click', () => {
    fileInput.click();
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (PNG, JPG, JPEG)');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        alert('File size exceeds 16MB limit');
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        predictBtn.disabled = false;
        results.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

predictBtn.addEventListener('click', () => {
    if (!selectedFile) return;

    predictBtn.disabled = true;
    loading.style.display = 'block';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('image', selectedFile);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        predictBtn.disabled = false;

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        displayResults(data);
    })
    .catch(error => {
        loading.style.display = 'none';
        predictBtn.disabled = false;
        alert('Error analyzing image: ' + error.message);
    });
});

function displayResults(data) {
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const probabilities = document.getElementById('probabilities');
    const treatmentSection = document.getElementById('treatmentSection');
    const treatmentText = document.getElementById('treatmentText');

    const isHealthy = data.is_healthy;

    resultCard.querySelector('.result-header').className =
        'result-header ' + (isHealthy ? 'healthy' : 'disease');

    resultIcon.textContent = isHealthy ? '\u2705' : '\u26A0\uFE0F';
    resultTitle.textContent = data.display_name;

    const confidencePercent = (data.confidence * 100).toFixed(1);
    confidenceBar.style.width = confidencePercent + '%';
    confidenceValue.textContent = confidencePercent + '%';

    probabilities.innerHTML = '';
    for (const [className, prob] of Object.entries(data.all_probabilities)) {
        const item = document.createElement('div');
        item.className = 'probability-item';

        const percent = (prob * 100).toFixed(1);
        item.innerHTML = `
            <span class="prob-label">${className.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${percent}%"></div>
            </div>
            <span class="prob-value">${percent}%</span>
        `;
        probabilities.appendChild(item);
    }

    if (isHealthy) {
        treatmentSection.style.display = 'none';
    } else {
        treatmentSection.style.display = 'block';
        treatmentText.textContent = data.treatment;
    }

    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
