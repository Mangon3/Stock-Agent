
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
          "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1",
          isUser ? "bg-blue-600" : "bg-purple-600"
        )}>
          {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-white" />}
        </div>

        {/* Bubble */}
        <div className={cn(
          "rounded-2xl px-5 py-3 text-sm md:text-base shadow-md",
          isUser 
            ? "bg-blue-600 text-white rounded-tr-none" 
            : "bg-gray-800 border border-gray-700 text-gray-100 rounded-tl-none w-full"
        )}>
          
          {/* Loading State */}
          {message.isLoading && <LoadingState />}

          {/* Text Content (if any) */}
          {message.content && (
            <p className="whitespace-pre-wrap">{message.content}</p>
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
