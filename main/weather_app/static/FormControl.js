document.getElementById('weather-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const algorithm = document.getElementById('algorithm').value;

    document.getElementById('algorithm').addEventListener('change', function() {

    // Xóa kết quả dự đoán và báo cáo cũ (nếu có)
    document.getElementById('result').innerHTML = '';


});

    const data = {
        precipitation: document.getElementById('precipitation').value,
        temp_max: document.getElementById('temp_max').value,
        temp_min: document.getElementById('temp_min').value,
        wind: document.getElementById('wind').value,
        algorithm: algorithm
    };
    console.log(data);

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerHTML = ` <h3 class="text-result"> Dự báo thời tiết: ${data.prediction}</h3> `;

    })
    .catch(error => console.error('Error:', error));
});
