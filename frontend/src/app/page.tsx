'use client';

import { useState, useEffect } from 'react';
import { Search, Sparkles, Settings, Key, ChevronDown, ChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { LoadingState } from '@/components/LoadingState';
import { AnalysisReport } from '@/components/AnalysisReport';

export default function Home() {
  const [symbol, setSymbol] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [data, setData] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState('');
  
  // Settings State
  const [apiKey, setApiKey] = useState('');
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('gemini_api_key');
    if (stored) setApiKey(stored);
  }, []);

  const handleKeyChange = (val: string) => {
    setApiKey(val);
    localStorage.setItem('gemini_api_key', val);
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol) return;

    setStatus('loading');
    setErrorMsg('');
    setData(null);

    try {
      const response = await fetch('https://mangonnn-stock-agent.hf.space/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-gemini-api-key': apiKey // Pass the key (can be empty, backend handles fallback)
        },
        body: JSON.stringify({ 
          symbol: symbol.toUpperCase(),
          timeframe_days: 7 // Default
        }),
      });

      if (!response.ok) {
        if (response.status === 429) {
           throw new Error("API Limit Reached (Free Tier). Please try again in ~60 seconds.");
        }
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      setData(result);
      setStatus('success');
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Something went wrong. Please try again.");
      setStatus('error');
    }
  };

  return (
    <main className="min-h-screen bg-black text-white selection:bg-blue-500/30">
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-[100px] opacity-50 mix-blend-screen animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-[100px] opacity-30 mix-blend-screen" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 py-12 md:py-24 space-y-12">
        

        {/* Header & Settings */}
        <div className="text-center space-y-4 relative">
            <div className="absolute top-0 right-0">
                <button 
                  onClick={() => setShowSettings(!showSettings)}
                  className="p-2 text-gray-500 hover:text-white transition-colors"
                >
                    <Settings className="w-6 h-6" />
                </button>
            </div>

           <AnimatePresence>
            {showSettings && (
              <motion.div 
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="bg-gray-900/80 border border-gray-800 rounded-xl p-6 max-w-md mx-auto mb-8 text-left space-y-3">
                  <div className="flex items-center space-x-2 text-sm font-medium text-gray-300">
                    <Key className="w-4 h-4 text-yellow-500" />
                    <span>Gemini API Key (Optional)</span>
                  </div>
                  <p className="text-xs text-gray-500">
                    Provide your own key to bypass server rate limits. Keys are stored locally in your browser.
                  </p>
                  <input 
                    type="password" 
                    value={apiKey}
                    onChange={(e) => handleKeyChange(e.target.value)}
                    placeholder="AIzaSy..."
                    className="w-full bg-black/50 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono text-white focus:ring-1 focus:ring-blue-500 outline-none"
                  />
                </div>
              </motion.div>
            )}
           </AnimatePresence>

          <div className="inline-flex items-center space-x-2 bg-gray-900/50 border border-gray-800 rounded-full px-4 py-1.5 mb-4 backdrop-blur-sm">
             <Sparkles className="w-4 h-4 text-blue-400" />
             <span className="text-xs font-medium text-gray-300">Powered by Gemini 2.5 & Finnhub</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">
            Stock Agent
          </h1>
          <p className="text-lg text-gray-400 max-w-xl mx-auto">
            Autonomous AI market analyst that combines macro news sentiment with micro-model technical predictions.
          </p>
        </div>

        {/* Search Input (Only show when not success to keep header clean, or kept always? Let's keep always but minimal) */}
        <div className={cn(
          "max-w-md mx-auto transition-all duration-500",
          status === 'success' ? "scale-90 opacity-75" : "scale-100"
        )}>
          <form onSubmit={handleAnalyze} className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl blur opacity-25 group-hover:opacity-50 transition duration-500" />
            <div className="relative flex items-center bg-gray-900 border border-gray-800 rounded-xl p-2 shadow-2xl">
              <Search className="w-5 h-5 text-gray-500 ml-3" />
              <input 
                type="text" 
                placeholder="Enter stock symbol (e.g. MSFT)" 
                className="w-full bg-transparent border-none focus:ring-0 text-white placeholder-gray-500 px-4 py-2"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
              />
              <button 
                type="submit"
                disabled={status === 'loading'}
                className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {status === 'loading' ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>
          </form>
          {errorMsg && (
            <p className="text-red-400 text-sm mt-3 text-center bg-red-900/10 border border-red-900/50 p-2 rounded-lg">
              {errorMsg}
            </p>
          )}
        </div>

        {/* Content Area */}
        <div className="min-h-[400px]">
          {status === 'loading' && <LoadingState />}
          {status === 'success' && data && <AnalysisReport data={data} />}
          {status === 'idle' && (
             <div className="text-center text-gray-600 mt-20">
                <p>Enter a symbol above to start the autonomous analysis agent.</p>
             </div>
          )}
        </div>

      </div>
    </main>
  );
}
