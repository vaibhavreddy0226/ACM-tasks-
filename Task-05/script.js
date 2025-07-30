// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getDatabase, ref, push } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";

// Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyBEfrPGyvXDlj6pjowtaQ9yqdjF0Ptr2ec",
  authDomain: "author-book-finder.firebaseapp.com",
  databaseURL: "https://author-book-finder-default-rtdb.firebaseio.com",
  projectId: "author-book-finder",
  storageBucket: "author-book-finder.firebasestorage.app",
  messagingSenderId: "1063298310379",
  appId: "1:1063298310379:web:08ea50bf1a3580698d9eae",
  measurementId: "G-FY1L20XGJF"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// DOM elements
const form = document.getElementById("authorForm");
const input = document.getElementById("authorInput");
const bookList = document.getElementById("bookList");
const resultsHeader = document.getElementById("resultsHeader");

// Submit handler
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const author = input.value.trim();
  if (!author) return;

  // Store author search to Firebase
  try {
    await push(ref(database, 'authorSearches'), { author: author, timestamp: Date.now() });
  } catch (err) {
    console.error("Firebase write error:", err);
  }

  // Fetch from OpenLibrary
  try {
    const response = await fetch(`https://openlibrary.org/search.json?author=${encodeURIComponent(author)}&limit=10`);
    const data = await response.json();

    bookList.innerHTML = ""; // Clear previous
    if (data.docs.length === 0) {
      bookList.innerHTML = `<tr><td colspan="3">No publications by this author.</td></tr>`;
      return;
    }

    // Display books
    data.docs.slice(0, 10).forEach((book, index) => {
      const title = book.title || "Unknown Title";
      const coverId = book.cover_i;
      const coverURL = coverId
        ? `https://covers.openlibrary.org/b/id/${coverId}-M.jpg`
        : null;

      const row = document.createElement("tr");

      row.innerHTML = `
        <td>${index + 1}</td>
        <td>${title}</td>
        <td>${coverURL ? `<img src="${coverURL}" alt="Cover" />` : "Cover not available"}</td>
      `;

      bookList.appendChild(row);
    });
  } catch (error) {
    console.error("Fetch error:", error);
    bookList.innerHTML = `<tr><td colspan="3">Failed to fetch data. Please try again later.</td></tr>`;
  }
});
