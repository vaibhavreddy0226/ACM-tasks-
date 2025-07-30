const API_KEY = "Vt3c03KOtm78IiuS4gf4b7WUJboBxxyoTgupL96Y";
const url = `https://api.nasa.gov/planetary/apod?api_key=${API_KEY}`;

fetch(url)
  .then(response => response.json())
  .then(data => {
    document.getElementById("title").textContent = data.title;
    document.getElementById("image").src = data.url;
    document.getElementById("explanation").textContent = data.explanation;
    document.getElementById("date").textContent = `Date: ${data.date}`;
  })
  