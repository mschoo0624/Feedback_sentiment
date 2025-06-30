"use client"; // Should be rendered on the client side.

import React, { useState } from 'react';
import axios from 'axios'; // to send HTTP requests.

import './Homepage.css'; // Import custom CSS for scrollbar, buttons, inputs

// Define interfaces for expected response structure
interface Summary {
  Like: number;
  Dislike: number;
  TopDislikeWord: string;
}

interface CommentDetail {
  text: string;
  sentiment: 'Like' | 'Dislike';
  confidence: number;
  is_sarcastic: boolean;
  was_translated: boolean;
}

interface AnalysisResult {
  summary: Summary;
  detailed: CommentDetail[];
}

export default function Homepage() {
  // The input field value (the URL pasted by the user).
  const [url, setUrl] = useState<string>('');

  // The result returned from the backend after analysis.
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  // Boolean to indicate if a request is in progress
  const [loading, setLoading] = useState<boolean>(false);

  // Stores any error message that occurs during the request.
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setAnalysisResult(null);
    setLoading(true);
    setError(null);

    try {
      // sends the "GET" request to the FastAPI backend. and passes the url as a query parameter.
      const request = await axios.get<AnalysisResult>(`http://localhost:8000/scrape-and-analyze?url=${encodeURIComponent(url)}`);
      setAnalysisResult(request.data);
    } catch (err: any) {
      console.error("Error during analysis:", err);
      if (err.response) {
        setError(err.response.data.detail || "An error occurred during analysis.");
      } else {
        setError("Network error or server is unreachable.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="homepage-wrapper">
      <div className="content-box">
        {/* Title */}
        <h1 className="title">
          FeedbackSentinel: Live Comment Analysis 🚀
        </h1>

        {/* Subheading */}
        <p className="subtitle">
          Paste any link with a comment section and see instant sentiment analysis of public comments using our Machine Learning model.
        </p>

        {/* Input + Button */}
        <div className="input-button-group">
          <input
            type="text"
            className="input-field"
            placeholder="Enter a URL with public comments (e.g., blog post, product review page)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => { // for enter key to work. 
              if (e.key === 'Enter') {
                handleAnalyze();
              }
            }}
          />
          <button
            onClick={handleAnalyze}
            className="analyze-btn"
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Comments'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-box" role="alert">
            <strong>Error!</strong> {error}
          </div>
        )}

        {/* Display Analysis Results */}
        {analysisResult && (
          <div className="results">
            {/* Summary Stats */}
            <h2>Summary</h2>
            <div className="summary-grid">
              <div className="summary-box like-box">
                <p>Likes:</p>
                <p className="summary-count">{analysisResult.summary.Like}</p>
              </div>
              <div className="summary-box dislike-box">
                <p>Dislikes:</p>
                <p className="summary-count">{analysisResult.summary.Dislike}</p>
              </div>
              <div className="summary-box word-box">
                <p>Most Common Word in Dislikes:</p>
                <p className="summary-word">{analysisResult.summary.TopDislikeWord || 'N/A'}</p>
              </div>
            </div>

            {/* Detailed Comments */}
            <h2>Detailed Comments</h2>
            <div className="comments-list">
              {analysisResult.detailed.map((comment, index) => (
                <div
                  key={index}
                  className={`comment-card ${comment.sentiment === 'Like' ? 'like' : 'dislike'}`}
                >
                  <p className="comment-text">{comment.text}</p>
                  <p className="comment-meta">
                    Sentiment: <strong>{comment.sentiment}</strong> | Confidence: <strong>{comment.confidence}</strong> | 
                    Sarcastic: <strong>{comment.is_sarcastic ? 'Yes' : 'No'}</strong> | 
                    Translated: <strong>{comment.was_translated ? 'Yes' : 'No'}</strong>
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
