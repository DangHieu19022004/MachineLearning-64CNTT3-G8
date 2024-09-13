document.getElementById('weather-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const data = {
        precipitation: document.getElementById('precipitation').value,
        temp_max: document.getElementById('temp_max').value,
        temp_min: document.getElementById('temp_min').value,
        wind: document.getElementById('wind').value,
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
        document.getElementById('result').innerText = `Dự báo: ${data.prediction}, Độ tin cậy: ${data.confidence * 100}%`;
    });
});
