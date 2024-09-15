document.getElementById('weather-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const algorithm = document.getElementById('algorithm').value;

//     document.getElementById('algorithm').addEventListener('change', function() {

//     // Xóa kết quả dự đoán và báo cáo cũ (nếu có)
//     document.getElementById('result').innerHTML = '';
//     document.getElementById('classification-report').innerHTML = '';
//     document.getElementById('confusion-matrix').src = '';
//     document.getElementById('entropy-diagram').src = '';

// });

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
        document.getElementById('result').innerHTML = ` <p class="text-result"> Dự báo thời tiết: ${data.prediction} </br> Độ tin cậy của thuật toán: ${data.confidence}% </p> `;

        // Display classification report as HTML table
        document.getElementById('classification-report').innerHTML = `<p class="text-result">Báo cáo phân loại:</p> </br> ${data.report}` ;

        // Display confusion matrix image
        document.getElementById('classificationrp').innerText = "Báo cáo phân loại: ";
        document.getElementById('confusion-matrix').src = 'data:image/png;base64,' + data.plot_url;

        // Display entropy diagram image
        document.getElementById('diagram').innerText = "Đồ thị entropy: ";
        document.getElementById('entropy-diagram').src = 'data:image/png;base64,' + data.entropy_url;
    })
    .catch(error => console.error('Error:', error));
});
