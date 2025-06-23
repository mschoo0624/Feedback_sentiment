import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';      // your global styles or Tailwind base imports
import App from './App';   // your main app component

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
