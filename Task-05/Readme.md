# 📚 Author Book Finder 🔍

This project is a simple web application that allows users to **search for books by author name** using the [Open Library API](https://openlibrary.org/developers/api) and display the **book titles along with their cover pages** in a responsive table format.

It also uses **Firebase Hosting** to deploy the project live to the web!

---

## 🚀 Features

- 🔎 Search for books by entering the author's name.
- 📖 Display the **first 10 books** with:
  - Serial Number (Index)
  - Title
  - Cover Page (if available; otherwise shows "Cover not available")


---

## 🧰 Tech Stack

- HTML5
- CSS3
- JavaScript (ES Modules)
- [Open Library Search API](https://openlibrary.org/dev/docs/api/search)
- [Firebase](https://firebase.google.com/) (for realtime database)

---

## 🔗 Live Demo

> You can access the live deployed project here:  
**[My webpage](https://vaibhavreddy0226.github.io/book-finder/)**  

---

## 🧪 How It Works

1. Enter an author's name in the search form.
2. On form submission:
   - The JavaScript script fetches data from the Open Library API.
   - It extracts up to the first 10 book titles and corresponding cover images.
   - Displays the data neatly in a responsive HTML table.
   - If no books are found, it shows a user-friendly message.

---

