'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Settings, Key, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage, Message } from '@/components/ChatMessage';
import { v4 as uuidv4 } from 'uuid';

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Settings State
  const [apiKey, setApiKey] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  
  // Auto-scroll ref
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = localStorage.getItem('gemini_api_key');
    if (stored) setApiKey(stored);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleKeyChange = (val: string) => {
    setApiKey(val);
    localStorage.setItem('gemini_api_key', val);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setIsLoading(true);

    // Add User Message
    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      content: userQuery
    };
    
    // Add Placeholder Bot Message
    const botMsgId = uuidv4();
    const botPlaceholder: Message = {
      id: botMsgId,
      role: 'assistant',
      isLoading: true
    };

    setMessages(prev => [...prev, userMsg, botPlaceholder]);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://mangonnn-stock-agent.hf.space';
      
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-gemini-api-key': apiKey
        },
        body: JSON.stringify({ 
          query: userQuery,
          timeframe_days: 7 
        }),
      });

      if (!response.ok) {
         if (response.status === 429) throw new Error("API Limit Reached. Please try again later.");
         throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();

      // Update Bot Message with Result
      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId 
          ? { ...msg, isLoading: false, data: result, content: result.final_report } 
          : msg
      ));

    } catch (err: any) {
      console.error(err);
      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId 
          ? { ...msg, isLoading: false, isError: true, content: err.message || "Something went wrong." } 
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex flex-col h-screen bg-black text-white selection:bg-blue-500/30 overflow-hidden">
      
      {/* Background Effects */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-[100px] opacity-30 animate-pulse" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-[100px] opacity-20" />
      </div>

      {/* Header */}
      <header className="flex-none z-10 border-b border-gray-800 bg-black/50 backdrop-blur-md p-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
             <Sparkles className="w-5 h-5 text-blue-400" />
             <h1 className="text-xl font-bold tracking-tight">Stock Agent <span className="text-gray-500 text-sm font-normal ml-2 hidden sm:inline">Autonomous Analyst</span></h1>
          </div>
          <button 
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-500 hover:text-white transition-colors hover:bg-gray-800 rounded-lg"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="flex-none z-20 bg-gray-900 border-b border-gray-800"
          >
            <div className="max-w-5xl mx-auto p-4">
                <div className="flex items-center space-x-2 text-sm font-medium text-gray-300 mb-2">
                  <Key className="w-4 h-4 text-yellow-500" />
                  <span>Gemini API Key</span>
                </div>
                <input 
                  type="password" 
                  value={apiKey}
                  onChange={(e) => handleKeyChange(e.target.value)}
                  placeholder="AIzaSy..."
                  className="w-full max-w-md bg-black/50 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto z-0 p-4 scroll-smooth">
        <div className="max-w-4xl mx-auto min-h-full flex flex-col justify-end">
           {messages.length === 0 ? (
             <div className="flex flex-col items-center justify-center h-full text-center space-y-4 py-20 opacity-50">
                <Sparkles className="w-12 h-12 text-blue-500" />
                <h2 className="text-2xl font-semibold">How can I help you today?</h2>
                <p className="max-w-md text-gray-400">Ask about any stock (e.g., "Analyze Microsoft" or "What's the outlook for NVDA?"). I'll gather news, run my models, and give you a report.</p>
             </div>
           ) : (
             messages.map(msg => (
               <ChatMessage key={msg.id} message={msg} />
             ))
           )}
           <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-none z-10 p-4 bg-gradient-to-t from-black via-black/90 to-transparent">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative flex items-center bg-gray-900 border border-gray-800 rounded-xl p-2 shadow-2xl focus-within:border-blue-500/50 transition-colors">
            <input 
              type="text" 
              placeholder="Ask about a stock..." 
              className="w-full bg-transparent border-none focus:ring-0 text-white placeholder-gray-500 px-4 py-3"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <button 
              type="submit"
              disabled={!input.trim() || isLoading}
              className="bg-blue-600 hover:bg-blue-500 text-white p-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <p className="text-center text-xs text-gray-600 mt-2">
            AI can make mistakes. Please verify important financial information.
          </p>
        </div>
      </div>

    </main>
  );
}
