
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content?: string;
  data?: any; 
  isLoading?: boolean;
  isError?: boolean;
  loadingProgress?: number;
  loadingMessage?: string;
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
          "rounded-md px-4 py-2 text-xs md:text-sm shadow-none overflow-hidden",
          isUser 
            ? "bg-white text-black rounded-tr-none" 
            : "bg-black border border-gray-800 text-gray-300 rounded-tl-none w-full"
        )}>
          
          {/* Loading State */}
          {message.isLoading && (
            <LoadingState 
              progress={message.loadingProgress} 
              message={message.loadingMessage} 
            />
          )}

          {/* Text Content (Markdown) */}
          {message.content && (
            <div className={cn("prose prose-sm max-w-none font-mono", isUser ? "prose-p:text-black" : "prose-invert")}>
               <ReactMarkdown
                  components={{
                    p: ({node, ...props}) => <p className={cn("mb-2 last:mb-0", isUser ? "text-black" : "text-gray-300")} {...props} />,
                    strong: ({node, ...props}) => <strong className={cn("font-bold", isUser ? "text-black" : "text-white")} {...props} />,
                    h1: ({node, ...props}) => <h1 className={cn("text-lg font-bold mt-4 mb-2 uppercase tracking-wider", isUser ? "text-black" : "text-white")} {...props} />,
                    h2: ({node, ...props}) => <h2 className={cn("text-base font-bold mt-4 mb-2 border-b border-gray-700 pb-1", isUser ? "text-black border-gray-300" : "text-white")} {...props} />,
                    h3: ({node, ...props}) => <h3 className={cn("text-sm font-bold mt-3 mb-1", isUser ? "text-black" : "text-white")} {...props} />,
                    ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2 space-y-1" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal pl-4 mb-2 space-y-1" {...props} />,
                    li: ({node, ...props}) => <li className={isUser ? "text-black" : "text-gray-300"} {...props} />,
                    code: ({node, ...props}) => <code className={cn("px-1 py-0.5 rounded text-xs", isUser ? "bg-gray-200 text-black" : "bg-gray-900 text-white")} {...props} />
                  }}
               >
                 {message.content}
               </ReactMarkdown>
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
