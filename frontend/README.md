# 🧠 FeedbackSentinel

> Live Public Comment Analyzer using FastAPI, Transformers, Web Scraping, and Next.js.

---

## 🚀 Overview
**FeedbackSentinel** is a powerful full-stack application that:
- Accepts **any URL** with a public comment section (e.g., product pages, blog posts, forums)
- **Scrapes** visible comments from the page
- Performs **real-time sentiment analysis** using a fine-tuned Machine Learning model
- Highlights the **most commonly mentioned word** in negative (Dislike) feedback
- Displays everything in a beautiful, responsive **dashboard UI** built with **React (Next.js) + Tailwind CSS**
---

## 🛠️ Project Structure
### 🧠 Backend: FastAPI + HuggingFace Transformers
- **ML Model**: Fine-tuned transformer (e.g. DistilBERT) to classify feedback as `Like` or `Dislike`
- **Endpoints**:
  - `POST /analyze`: Analyze a single feedback text
  - `GET /scrape-and-analyze?url=<url>`: Scrape public comments from any URL and analyze all of them
- **Features**:
  - Sarcasm detection
  - Translation support for non-English comments
  - Most frequent Dislike word insight
  - Auto-save to MySQL database (SQLAlchemy ORM)

### 🌐 Frontend: React + Tailwind + Next.js
- Input a **public URL**
- Display:
  - Total Likes 👍 and Dislikes 👎
  - Top keyword in negative comments 🧩
  - Scrollable list of analyzed comments (with sarcasm + translation flags)
---
- Install axios (to send HTTP requests)

## ⚙️ Installation & Setup
### ✅ Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

# Make sure MySQL is running and configured in database.py
uvicorn app.main:app --reload
