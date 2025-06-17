# 🔍 FeedbackSentinel Project.

FeedbackSentinel is an AI-powered sentiment analysis tool that classifies customer feedback into "Like" or "Dislike" categories. It uses a fine-tuned language model (like DistilBERT or RoBERTa) for fast, accurate classification and handles sarcasm or ambiguous comments using GPT-4.

The tool offers a FastAPI backend for processing text and a React dashboard to visualize results in real time. It can also analyze comments from 
website links and integrate with tools like Slack or a Chrome extension.

Use cases: Improve customer support, track product sentiment, and enhance marketing insights.

# 🧠 Features
## 🔍 Binary Sentiment Classification
Classify text as Like or Dislike, with confidence scores.

## 🧾 Context-Aware Analysis
Handles sarcasm, idioms, and cultural nuance using GPT-4 fallback for ambiguous inputs.

## 🏷️ Custom Keyword Mapping
Let businesses define industry-specific sentiment terms (e.g., “quiet” is positive for hotels).

## 📊 Real-Time Dashboard
Track trends, visualize sentiment over time, and see top keywords from customer feedback.

## 🌐 Website Comment Analyzer
Paste any link with a comment section and see instant sentiment analysis of public comments.

## 🔌 Integrations
Chrome extension and Slackbot for in-the-moment sentiment classification.

# 📈 Impact & ROI
⏱️ Review Time - Benefit: 80% reduction in manual feedback review
🎯 Customer Experience - Benefit: Faster response to negative feedback
📣 Marketing - Benefit: Amplify positive customer feedback

# ✅ STAGE 1: BACKEND DEVELOPMENT (FastAPI + Transformers)
## 🛠️ Goal: Build a FastAPI server that exposes a /classify endpoint to accept feedback text and return sentiment.


# 📌 TODO / Future Features
- Feedback dashboard (with Plotly/Dash)
- User auth & history tracking
- Admin rule-based sentiment overrides
- Secondary language output
- WebSocket real-time feedback monitor
