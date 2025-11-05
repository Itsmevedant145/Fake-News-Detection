import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, Sparkles, Copy, Check } from 'lucide-react';

const FakeNewsDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const checkNews = async () => {
    setResult(null);
    setError('');

    if (!text.trim()) {
      setError('Please enter some news text or URL');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (response.ok) {
        const isFake = data.prediction === 'Fake';
        setResult({
          verdict: isFake ? 'Not true' : 'Verified',
          isFake: isFake,
          reason: data.reason || (isFake 
            ? 'Our AI model has detected patterns commonly associated with fake news or misleading information in this content.'
            : 'The content appears to be credible. Our analysis shows patterns consistent with legitimate news sources.'),
          confidence: data.confidence || '95/100'
        });
      } else {
        setError(data.error || 'Something went wrong.');
      }
    } catch (err) {
      setError('⚠️ Could not connect to the server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const copyResult = () => {
    if (result) {
      const resultText = `Analysis Result:\n\nVerdict: ${result.verdict}\nReason: ${result.reason}\nConfidence: ${result.confidence}`;
      navigator.clipboard.writeText(resultText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-overlay filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-overlay filter blur-3xl opacity-20 animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-indigo-500 rounded-full mix-blend-overlay filter blur-3xl opacity-20 animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>

      {/* Floating particles */}
      <div className="absolute inset-0 pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full opacity-30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${5 + Math.random() * 10}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 5}s`
            }}
          ></div>
        ))}
      </div>

      <div className="relative w-full max-w-5xl z-10">
        {/* Main Card */}
        <div className="bg-slate-900/40 backdrop-blur-2xl rounded-3xl shadow-2xl border border-white/10 p-8 md:p-12 transition-all duration-500 hover:shadow-blue-500/20 hover:shadow-3xl">
          
          {/* Header Section */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-600 rounded-3xl mb-6 shadow-2xl shadow-blue-500/50 relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-400 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <Shield className="w-12 h-12 text-white relative z-10 transform group-hover:scale-110 transition-transform duration-300" strokeWidth={2.5} />
            </div>
            
            <h1 className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-200 via-purple-200 to-pink-200 mb-4 tracking-tight">
              Paste a news article URL or type a fact below and<br />
              <span className="text-4xl md:text-5xl">let our AI tell you if it's credible — or complete nonsense.</span>
            </h1>
          </div>

          {/* Input Section */}
          <div className="mb-8">
            <div className="relative">
              <input
                type="text"
                placeholder="Enter news text or URL..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && checkNews()}
                className="w-full px-6 py-5 text-lg text-white placeholder-slate-400 bg-slate-800/50 backdrop-blur-sm border-2 border-slate-600 rounded-2xl transition-all duration-300 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 focus:outline-none focus:bg-slate-800/70"
              />
              <button
                onClick={checkNews}
                disabled={loading || !text.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 px-8 py-3 bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white font-bold rounded-xl shadow-lg shadow-emerald-500/30 transition-all duration-300 disabled:from-slate-600 disabled:to-slate-700 disabled:shadow-none disabled:cursor-not-allowed hover:shadow-xl hover:scale-105 active:scale-95 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Analyzing
                  </>
                ) : (
                  <>
                    🔍 Analyze
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Section */}
          {result && (
            <div className="mt-8 p-8 bg-slate-800/60 backdrop-blur-xl rounded-3xl border-2 border-slate-700 shadow-2xl animate-slideIn">
              <div className="flex items-center gap-3 mb-6">
                <div className="flex items-center justify-center w-12 h-12 bg-slate-700/50 rounded-xl">
                  <span className="text-2xl">📋</span>
                </div>
                <h2 className="text-2xl font-bold text-white">Analysis Result:</h2>
              </div>

              <div className="space-y-6">
                {/* Verdict */}
                <div className="flex items-start gap-4">
                  <div className="flex items-center justify-center w-10 h-10 bg-emerald-500/20 rounded-lg flex-shrink-0">
                    {result.isFake ? (
                      <AlertTriangle className="w-6 h-6 text-red-400" strokeWidth={2.5} />
                    ) : (
                      <CheckCircle className="w-6 h-6 text-emerald-400" strokeWidth={2.5} />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-emerald-400 font-bold text-lg">● Verdict:</span>
                    </div>
                    <p className={`text-xl font-semibold ${result.isFake ? 'text-red-300' : 'text-emerald-300'}`}>
                      {result.verdict}
                    </p>
                  </div>
                </div>

                {/* Reason */}
                <div className="flex items-start gap-4">
                  <div className="flex items-center justify-center w-10 h-10 bg-purple-500/20 rounded-lg flex-shrink-0">
                    <span className="text-xl">💬</span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-purple-400 font-bold text-lg">● Reason:</span>
                    </div>
                    <p className="text-slate-300 text-base leading-relaxed">
                      {result.reason}
                    </p>
                  </div>
                </div>

                {/* Confidence */}
                <div className="flex items-start gap-4">
                  <div className="flex items-center justify-center w-10 h-10 bg-blue-500/20 rounded-lg flex-shrink-0">
                    <span className="text-xl">📊</span>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-blue-400 font-bold text-lg">● Confidence:</span>
                    </div>
                    <p className="text-white text-xl font-bold">
                      {result.confidence}
                    </p>
                  </div>
                </div>
              </div>

              {/* Copy Button */}
              <div className="mt-8 flex justify-start">
                <button
                  onClick={copyResult}
                  className="px-6 py-3 bg-slate-700/50 hover:bg-slate-600/50 text-white font-semibold rounded-xl border border-slate-600 transition-all duration-300 flex items-center gap-2 hover:scale-105 active:scale-95"
                >
                  {copied ? (
                    <>
                      <Check className="w-5 h-5" />
                      Copied!
                    </>
                  ) : (
                    <>
                      📋 Copy Result
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-8 p-6 bg-amber-900/40 backdrop-blur-xl border-2 border-amber-600/50 rounded-2xl shadow-lg animate-slideIn">
              <div className="flex items-center gap-4">
                <div className="flex items-center justify-center w-12 h-12 bg-amber-500/20 rounded-full flex-shrink-0">
                  <AlertTriangle className="w-6 h-6 text-amber-400" strokeWidth={2.5} />
                </div>
                <p className="text-amber-200 font-medium">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Footer Note */}
        <p className="text-center text-slate-400 text-sm mt-8 px-4 flex items-center justify-center gap-2">
          <Sparkles className="w-4 h-4 text-yellow-400" />
          Powered by AI • For educational purposes • Always verify from multiple sources
          <Sparkles className="w-4 h-4 text-yellow-400" />
        </p>
      </div>

      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slideIn {
          animation: slideIn 0.5s ease-out;
        }
        @keyframes float {
          0%, 100% {
            transform: translateY(0) translateX(0);
          }
          25% {
            transform: translateY(-20px) translateX(10px);
          }
          50% {
            transform: translateY(-10px) translateX(-10px);
          }
          75% {
            transform: translateY(-15px) translateX(5px);
          }
        }
      `}</style>
    </div>
  );
};

export default FakeNewsDetector;