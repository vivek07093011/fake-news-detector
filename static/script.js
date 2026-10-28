async function checkNews() {

    const newsText = document.getElementById("newsText").value;

    const resultDiv = document.getElementById("result");
    const prediction = document.getElementById("prediction");
    const confidence = document.getElementById("confidence");
    const loading = document.getElementById("loading");

    if (!newsText.trim()) {
        alert("Please enter some news text.");
        return;
    }

    loading.style.display = "block";
    resultDiv.style.display = "none";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                news: newsText
            })
        });

        const data = await response.json();

        loading.style.display = "none";

        prediction.innerText = "Result: " + data.prediction;
        confidence.innerText = data.confidence + "%";

        resultDiv.style.display = "block";

    } catch (error) {

        loading.style.display = "none";

        alert("Something went wrong!");
        console.error(error);
    }
}
