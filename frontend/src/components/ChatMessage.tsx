
import { User, Bot } from 'lucide-react';
import { cn } from '@/lib/utils';
import { AnalysisReport } from '@/components/AnalysisReport';
import { LoadingState } from '@/components/LoadingState';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content?: string;
  data?: any; // The full analysis JSON
  isLoading?: boolean;
  isError?: boolean;
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={cn(
      "flex w-full mb-6",
      isUser ? "justify-end" : "justify-start"
    )}>
      <div className={cn(
        "flex max-w-[90%] md:max-w-[80%] gap-3",
        isUser ? "flex-row-reverse" : "flex-row"
      )}>
        {/* Avatar */}
        <div className={cn(
          "w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-1 border border-gray-800",
          isUser ? "bg-white text-black" : "bg-black text-white"
        )}>
          {isUser ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
        </div>

        {/* Bubble */}
        <div className={cn(
          "rounded-md px-4 py-2 text-xs md:text-sm shadow-none",
          isUser 
            ? "bg-white text-black rounded-tr-none" 
            : "bg-black border border-gray-800 text-gray-300 rounded-tl-none w-full"
        )}>
          
          {/* Loading State */}
          {message.isLoading && <LoadingState />}

          {/* Text Content (if any) */}
          {message.content && (
            <p className="whitespace-pre-wrap font-mono">{message.content}</p>
          )}

          {/* Analysis Report (if data exists) */}
          {message.data && (
            <div className="mt-4">
               <AnalysisReport data={message.data} />
            </div>
          )}

          {/* Error State */}
          {message.isError && (
            <div className="text-red-400 mt-2 bg-red-900/20 p-2 rounded border border-red-900/50">
              Error: {message.content}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
