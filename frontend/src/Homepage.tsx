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
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-4xl container">
        {/* Title */}
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">
          FeedbackSentinel: Live Comment Analysis 🚀
        </h1>

        {/* Subheading */}
        <p className="text-center text-gray-600 mb-8">
          Paste any link with a comment section and see instant sentiment analysis of public comments using our Machine Learning model.
        </p>

        {/* Input + Button */}
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <input
            type="text"
            className="flex-grow p-3 border border-gray-300 rounded-lg input-focus-glow focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter a URL with public comments (e.g., blog post, product review page)"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button
            onClick={handleAnalyze}
            className={`px-6 py-3 rounded-lg text-white font-semibold btn-animate transition-colors duration-200 ${
              loading ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Comments'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}

        {/* Display Analysis Results */}
        {analysisResult && (
          <div className="mt-8">
            {/* Summary Stats */}
            <h2 className="text-2xl font-bold text-gray-700 mb-4">Summary</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-green-50 p-4 rounded-lg shadow-lg">
                <p className="text-lg font-semibold text-green-700">Likes:</p>
                <p className="text-3xl font-bold text-green-600">{analysisResult.summary.Like}</p>
              </div>
              <div className="bg-red-50 p-4 rounded-lg shadow-lg">
                <p className="text-lg font-semibold text-red-700">Dislikes:</p>
                <p className="text-3xl font-bold text-red-600">{analysisResult.summary.Dislike}</p>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg shadow-lg">
                <p className="text-lg font-semibold text-yellow-700">Most Common Word in Dislikes:</p>
                <p className="text-xl font-bold text-yellow-600">
                  {analysisResult.summary.TopDislikeWord || 'N/A'}
                </p>
              </div>
            </div>

            {/* Detailed Comments */}
            <h2 className="text-2xl font-bold text-gray-700 mb-4">Detailed Comments</h2>
            <div className="space-y-4 max-h-96 overflow-y-auto pr-2 scrollbar-thin">
              {analysisResult.detailed.map((comment, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg shadow ${
                    comment.sentiment === 'Like'
                      ? 'bg-green-50 border-l-4 border-green-500'
                      : 'bg-red-50 border-l-4 border-red-500'
                  }`}
                >
                  <p className="font-medium text-gray-800 mb-2">{comment.text}</p>
                  <p className="text-sm text-gray-600">
                    Sentiment:{' '}
                    <span className={`font-semibold ${comment.sentiment === 'Like' ? 'text-green-600' : 'text-red-600'}`}>
                      {comment.sentiment}
                    </span>{' '}
                    | Confidence: <span className="font-semibold">{comment.confidence}</span> | Sarcastic:{' '}
                    <span className="font-semibold">{comment.is_sarcastic ? 'Yes' : 'No'}</span> | Translated:{' '}
                    <span className="font-semibold">{comment.was_translated ? 'Yes' : 'No'}</span>
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