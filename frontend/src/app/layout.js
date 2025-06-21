import './global.css'; 

export const metadata = {
  title: 'FeedbackSentinel', // Good idea to change this title
  description: 'Live Comment Sentiment Analysis',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
