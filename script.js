document.addEventListener("DOMContentLoaded", function() {
    // Adjust the CSV file URL according to your server setup
    const csvUrl = '/prediction_files/predictions5min.csv';

    fetch(csvUrl)
    .then(response => response.text())
    .then(data => {
        let result = parseCSV(data);
        displayPredictions(result);
    })
    .catch(error => console.error('Error fetching the CSV file:', error));
});

function parseCSV(csvData) {
    let lines = csvData.split("\n");
    return lines.map(line => line.split(","));
}

function displayPredictions(data) {
    let predictionsDiv = document.getElementById('predictions');
    let htmlContent = "<ul>";
    data.forEach((row, index) => {
        if (index > 0) { // Skip header row
            htmlContent += `<li>${row.join(' - ')}</li>`;
        }
    });
    htmlContent += "</ul>";
    predictionsDiv.innerHTML = htmlContent;
}
