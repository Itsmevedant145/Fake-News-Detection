import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, Sparkles, TrendingUp, Globe, Tag } from 'lucide-react';

const FakeNewsDetector = () => {
  const [text, setText] = useState('');
  const [web, setWeb] = useState('');
  const [category, setCategory] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const categories = ['COVID-19', 'ELECTION', 'POLITICS', 'TERROR', 'VIOLENCE', 'SPORTS', 'ENTERTAINMENT', 'HEALTH', 'RELIGION'];

  const checkNews = async () => {
    setResult(null);
    setError('');

    if (!text.trim()) {
      setError('Please enter some news text');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: text.trim(),
          web: web.trim() || undefined,
          category: category || undefined
        })
      });

      const data = await response.json();

      if (response.ok) {
        const isFake = data.prediction === 'Fake';
        setResult({
          verdict: isFake ? 'Fake News' : 'Real News',
          isFake: isFake,
          reason: data.reason,
          confidence: data.confidence,
          confidence_percentage: data.confidence_percentage,
          model_votes: data.model_votes,
          model_agreement: data.model_agreement,
          analysis_notes: data.analysis_notes || [],
          indian_context: data.indian_context || {}
        });
      } else {
        setError(data.error || 'Something went wrong.');
      }
    } catch (err) {
      setError('⚠️ Could not connect to the server. Please ensure the backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const getCredibilityColor = (credibility) => {
    if (credibility === 'TRUSTED') return 'text-emerald-400';
    if (credibility === 'SUSPICIOUS') return 'text-red-400';
    return 'text-yellow-400';
  };

  const getCredibilityBg = (credibility) => {
    if (credibility === 'TRUSTED') return 'bg-emerald-500/20';
    if (credibility === 'SUSPICIOUS') return 'bg-red-500/20';
    return 'bg-yellow-500/20';
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500 rounded-full mix-blend-overlay filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-overlay filter blur-3xl opacity-20 animate-pulse" style={{animationDelay: '1s'}}></div>
      </div>

      <div className="relative w-full max-w-6xl z-10">
        <div className="bg-slate-900/40 backdrop-blur-2xl rounded-3xl shadow-2xl border border-white/10 p-8 md:p-12">
          
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-600 rounded-3xl mb-4 shadow-2xl">
              <Shield className="w-10 h-10 text-white" strokeWidth={2.5} />
            </div>
            
            <h1 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-200 via-purple-200 to-pink-200 mb-2">
              🇮🇳 Indian Fake News Detector
            </h1>
            <p className="text-slate-300 text-lg">Powered by IFND Dataset • AI-Enhanced Detection</p>
          </div>

          {/* Input Section */}
          <div className="space-y-4 mb-6">
            {/* News Text */}
            <div>
              <label className="block text-slate-300 font-semibold mb-2 text-sm">
                📰 News Statement *
              </label>
              <textarea
                placeholder="Enter news headline or article text..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="w-full px-4 py-3 text-white placeholder-slate-400 bg-slate-800/50 border-2 border-slate-600 rounded-xl transition-all duration-300 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 focus:outline-none resize-none"
              />
            </div>

            {/* Source and Category Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Source */}
              <div>
                <label className="block text-slate-300 font-semibold mb-2 text-sm">
                  <Globe className="w-4 h-4 inline mr-1" />
                  Source (optional)
                </label>
                <input
                  type="text"
                  placeholder="e.g., Times of India, NDTV, Unknown"
                  value={web}
                  onChange={(e) => setWeb(e.target.value)}
                  className="w-full px-4 py-3 text-white placeholder-slate-400 bg-slate-800/50 border-2 border-slate-600 rounded-xl transition-all duration-300 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 focus:outline-none"
                />
              </div>

              {/* Category */}
              <div>
                <label className="block text-slate-300 font-semibold mb-2 text-sm">
                  <Tag className="w-4 h-4 inline mr-1" />
                  Category (optional)
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-4 py-3 text-white bg-slate-800/50 border-2 border-slate-600 rounded-xl transition-all duration-300 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 focus:outline-none cursor-pointer"
                >
                  <option value="">Select category</option>
                  {categories.map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={checkNews}
              disabled={loading || !text.trim()}
              className="w-full px-8 py-4 bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 text-white font-bold text-lg rounded-xl shadow-lg shadow-emerald-500/30 transition-all duration-300 disabled:from-slate-600 disabled:to-slate-700 disabled:shadow-none disabled:cursor-not-allowed hover:scale-105 active:scale-95 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  Analyzing with AI...
                </>
              ) : (
                <>
                  🔍 Analyze News
                </>
              )}
            </button>
          </div>

          {/* Results */}
          {result && (
            <div className="mt-6 p-6 bg-slate-800/60 backdrop-blur-xl rounded-2xl border-2 border-slate-700 animate-slideIn">
              
              {/* Verdict Header */}
              <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-700">
                <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${result.isFake ? 'bg-red-500/20' : 'bg-emerald-500/20'}`}>
                  {result.isFake ? (
                    <AlertTriangle className="w-7 h-7 text-red-400" strokeWidth={2.5} />
                  ) : (
                    <CheckCircle className="w-7 h-7 text-emerald-400" strokeWidth={2.5} />
                  )}
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">{result.verdict}</h2>
                  <p className="text-slate-400 text-sm">Confidence: {result.confidence}</p>
                </div>
              </div>

              {/* Main Reason */}
              <div className="mb-6 p-4 bg-slate-700/30 rounded-xl">
                <p className="text-slate-200 leading-relaxed">{result.reason}</p>
              </div>

              {/* Indian Context Analysis */}
              {result.indian_context && (
                <div className="mb-6">
                  <h3 className="text-white font-bold mb-3 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-purple-400" />
                    Indian Context Analysis
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {/* Source Credibility */}
                    <div className={`p-3 ${getCredibilityBg(result.indian_context.source_credibility)} rounded-lg border border-slate-600`}>
                      <div className="text-xs text-slate-400 mb-1">Source Credibility</div>
                      <div className={`font-bold ${getCredibilityColor(result.indian_context.source_credibility)}`}>
                        {result.indian_context.source_credibility}
                      </div>
                    </div>

                    {/* Sensationalism */}
                    <div className="p-3 bg-blue-500/20 rounded-lg border border-slate-600">
                      <div className="text-xs text-slate-400 mb-1">Sensationalism Score</div>
                      <div className="font-bold text-blue-400">
                        {result.indian_context.sensationalism_score}/10
                      </div>
                    </div>

                    {/* Implausible Claims */}
                    <div className={`p-3 ${result.indian_context.has_implausible_claims ? 'bg-orange-500/20' : 'bg-emerald-500/20'} rounded-lg border border-slate-600`}>
                      <div className="text-xs text-slate-400 mb-1">Implausible Claims</div>
                      <div className={`font-bold ${result.indian_context.has_implausible_claims ? 'text-orange-400' : 'text-emerald-400'}`}>
                        {result.indian_context.has_implausible_claims ? 'YES ⚠️' : 'NO ✓'}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Votes */}
              {result.model_votes && (
                <div className="mb-6">
                  <h3 className="text-white font-bold mb-3">🤖 Model Ensemble Votes</h3>
                  <div className="space-y-2">
                    {Object.entries(result.model_votes).map(([model, vote]) => (
                      <div key={model} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                        <span className="text-slate-300">{model}</span>
                        <span className={`font-bold px-3 py-1 rounded-full text-sm ${vote === 'Fake' ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                          {vote}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="text-slate-400 text-sm mt-2">
                    Model Agreement: {result.model_agreement}
                  </p>
                </div>
              )}

              {/* Analysis Notes */}
              {result.analysis_notes && result.analysis_notes.length > 0 && (
                <div>
                  <h3 className="text-white font-bold mb-3">📝 Analysis Notes</h3>
                  <ul className="space-y-2">
                    {result.analysis_notes.map((note, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                        <span className="text-blue-400 mt-1">•</span>
                        <span>{note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-6 p-4 bg-amber-900/40 border-2 border-amber-600/50 rounded-xl animate-slideIn">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-6 h-6 text-amber-400" />
                <p className="text-amber-200">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-center text-slate-400 text-sm mt-6 flex items-center justify-center gap-2">
          <Sparkles className="w-4 h-4 text-yellow-400" />
          IFND Dataset (2013-2021) • 3-Model Ensemble • Indian Media Context
          <Sparkles className="w-4 h-4 text-yellow-400" />
        </p>
      </div>

      <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slideIn {
          animation: slideIn 0.5s ease-out;
        }
      `}</style>
    </div>
  );
};

export default FakeNewsDetector;