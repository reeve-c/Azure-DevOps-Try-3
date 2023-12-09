function sendNumbers() {
    // Get input values from the form
    var sl = document.getElementById("sepalLength").value;
    var sw = document.getElementById("sepalWidth").value;
    var pl = document.getElementById("petalLength").value;
    var pw = document.getElementById("petalWidth").value;

    // Send the numbers to app.py using fetch
    fetch('/multiply', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            flower_in: [sl,sw,pl,pw]
        }),
    })
    .then(response => response.json())
    .then(data => {
        // Display the result on the HTML page
        document.getElementById("result").innerText = "Species: " + data.result;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
