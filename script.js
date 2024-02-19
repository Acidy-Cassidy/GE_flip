document.addEventListener("DOMContentLoaded", function() {
    // Define all CSV file URLs
    const csvFiles = [
        '/prediction_files/predictions5min.csv',
        '/prediction_files/predictions5minH.csv',
        '/prediction_files/predictions5minL.csv',
        '/prediction_files/predictions10minC.csv',
        '/prediction_files/predictions10minH.csv',
        '/prediction_files/predictions10minL.csv'
    ];

    // Loop through each CSV file URL and fetch the data
    csvFiles.forEach(csvUrl => {
        fetch(csvUrl)
        .then(response => response.text())
        .then(data => {
            let result = parseCSV(data);
            displayPredictions(result, csvUrl);
        })
        .catch(error => console.error(`Error fetching the CSV file ${csvUrl}:`, error));
    });
});

function parseCSV(csvData) {
    let lines = csvData.split("\n");
    return lines.map(line => line.split(","));
}

function displayPredictions(data, csvUrl) {
    let predictionsDiv = document.getElementById('predictions');
    // Extract file name from csvUrl for display
    let fileName = csvUrl.split('/').pop();

    // Add a header for each CSV file
    let htmlContent = `<h2>${fileName}</h2><ul>`;

    data.forEach((row, index) => {
        if (index > 0) { // Skip header row
            htmlContent += `<li>${row.join(' - ')}</li>`;
        }
    });
    htmlContent += "</ul>";

    // Append the current CSV data to the predictionsDiv
    predictionsDiv.innerHTML += htmlContent;
}
