document.getElementById('weather-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const algorithm = document.getElementById('algorithm').value;

    const data = {
        precipitation: document.getElementById('precipitation').value,
        temp_max: document.getElementById('temp_max').value,
        temp_min: document.getElementById('temp_min').value,
        wind: document.getElementById('wind').value,
        algorithm: algorithm
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Dự báo: ${data.prediction}, Độ tin cậy: ${parseFloat(data.confidence.toFixed(2))}%`;

        // Display classification report as HTML table
        document.getElementById('classification-report').innerHTML = data.report;

        // Display confusion matrix image
        document.getElementById('confusion-matrix').src = 'data:image/png;base64,' + data.plot_url;
    })
    .catch(error => console.error('Error:', error));
});
