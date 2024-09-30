document.getElementById('stepThroughBtn').addEventListener('click', stepThroughKMeans);
document.getElementById('runToConvergenceBtn').addEventListener('click', runToConvergence);
document.getElementById('generateDatasetBtn').addEventListener('click', generateNewDataset);
document.getElementById('resetBtn').addEventListener('click', resetAlgorithm);

let canvas = document.getElementById('plotCanvas');
let ctx = canvas.getContext('2d');
let k = parseInt(document.getElementById('numClusters').value);
let method = document.getElementById('initMethod').value;
let manualCentroids = [];
let data = [];
let currentCentroids = null;
let hasConverged = false;
let manualSelectionComplete = false;

const scale = 60;
const xOffset = canvas.width / 2;
const yOffset = canvas.height / 2;

function generateNewDataset() {
    fetch('/api/generate_data')
        .then(response => response.json())
        .then(responseData => {
            data = responseData.data;
            resetAlgorithm();
            drawInitialPointsAndAxes();
        })
        .catch(error => console.error("Error fetching data:", error));
}

generateNewDataset();

canvas.addEventListener('click', function (event) {
    method = document.getElementById('initMethod').value;
    k = parseInt(document.getElementById('numClusters').value);

    if (method === 'manual' && manualCentroids.length < k) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const scaledX = (x - xOffset) / scale;
        const scaledY = -(y - yOffset) / scale;

        manualCentroids.push([scaledX, scaledY]);
        drawManualCentroid(scaledX, scaledY);

        if (manualCentroids.length === k) {
            currentCentroids = manualCentroids;
            manualSelectionComplete = true;
        }
    }
});

function resetAlgorithm() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    manualCentroids = [];
    currentCentroids = null;
    hasConverged = false;
    manualSelectionComplete = false;
    drawInitialPointsAndAxes();
}

function drawAxes() {
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(0, yOffset);
    ctx.lineTo(canvas.width, yOffset);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(xOffset, 0);
    ctx.lineTo(xOffset, canvas.height);
    ctx.stroke();

    ctx.font = '12px Arial';
    ctx.fillStyle = '#000';

    for (let i = -3; i <= 3; i++) {
        const xPos = i * scale + xOffset;
        ctx.fillText(i, xPos, yOffset + 15);
        ctx.beginPath();
        ctx.moveTo(xPos, yOffset - 5);
        ctx.lineTo(xPos, yOffset + 5);
        ctx.stroke();
    }

    for (let i = -3; i <= 3; i++) {
        const yPos = yOffset - i * scale;
        ctx.fillText(i, xOffset + 10, yPos + 5);
        ctx.beginPath();
        ctx.moveTo(xOffset - 5, yPos);
        ctx.lineTo(xOffset + 5, yPos);
        ctx.stroke();
    }
}

function drawInitialPointsAndAxes() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawAxes();

    for (let i = 0; i < data.length; i++) {
        ctx.fillStyle = '#3498db';
        ctx.beginPath();
        ctx.arc(data[i][0] * scale + xOffset, -data[i][1] * scale + yOffset, 5, 0, Math.PI * 2, true);
        ctx.fill();
    }
}

function drawManualCentroid(x, y) {
    drawCentroid(x, y);
}

function drawCentroid(x, y) {
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x * scale + xOffset - 10, -y * scale + yOffset - 10);
    ctx.lineTo(x * scale + xOffset + 10, -y * scale + yOffset + 10);
    ctx.moveTo(x * scale + xOffset + 10, -y * scale + yOffset - 10);
    ctx.lineTo(x * scale + xOffset - 10, -y * scale + yOffset + 10);
    ctx.stroke();
}

function stepThroughKMeans() {
    if (hasConverged) {
        alert("KMeans has already converged.");
        return;
    }

    method = document.getElementById('initMethod').value;
    k = parseInt(document.getElementById('numClusters').value);

    if (method === 'manual' && !manualSelectionComplete) {
        alert("Please select all centroids manually before stepping through KMeans.");
        return;
    }

    let bodyContent = {
        data: data,
        k: k,
        method: method,
        centroids: method === 'manual' ? currentCentroids || manualCentroids : currentCentroids
    };


    fetch('/api/step_kmeans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyContent)
    })
        .then(response => response.json())
        .then(step => {

            drawClusters(step.centroids, step.assignment);

            if (JSON.stringify(currentCentroids) === JSON.stringify(step.centroids)) {
                hasConverged = true;
                alert("KMeans has converged.");
            }

            currentCentroids = step.centroids;
        })
        .catch(error => console.error("Error fetching KMeans step:", error));
}

function drawClusters(centroids, assignment) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();

    for (let i = 0; i < data.length; i++) {
        const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c','#e67e22'];
        ctx.fillStyle = colors[assignment[i] % colors.length];
        ctx.beginPath();
        ctx.arc(data[i][0] * scale + xOffset, -data[i][1] * scale + yOffset, 5, 0, Math.PI * 2, true);
        ctx.fill();
    }

    if (centroids && centroids.length > 0) {
        centroids.forEach((centroid, index) => {
            drawCentroid(centroid[0], centroid[1]);
        });
    }
}

function runToConvergence() {
    method = document.getElementById('initMethod').value;
    k = parseInt(document.getElementById('numClusters').value);

    if (method === 'manual' && !manualSelectionComplete) {
        alert("Please select all centroids manually before running to convergence.");
        return;
    }

    let bodyContent = {
        data: data,
        k: k,
        method: method,
        centroids: method === 'manual' ? currentCentroids || manualCentroids : currentCentroids
    };

    fetch('/api/run_to_convergence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bodyContent)
    })
        .then(response => response.json())
        .then(convergenceResult => {
            drawClusters(convergenceResult.centroids, convergenceResult.assignment);

            hasConverged = true;
            alert("KMeans has converged.");
        })
        .catch(error => console.error("Error running KMeans to convergence:", error));
}