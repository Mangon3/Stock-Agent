import { motion } from 'framer-motion';

interface LoadingStateProps {
  progress?: number;
  message?: string;
}

export function LoadingState({ progress = 0, message = "Processing..." }: LoadingStateProps) {
  return (
    <div className="flex flex-col justify-center py-2 w-full min-w-[200px] max-w-[300px] space-y-2">
       {/* Progress Bar Container */}
       <div className="relative w-full h-1 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
         <motion.div 
           className="absolute top-0 left-0 h-full bg-white"
           initial={{ width: 0 }}
           animate={{ width: `${progress}%` }}
           transition={{ duration: 0.5, ease: "easeInOut" }}
         />
       </div>
       
       {/* Progress Message */}
       <div className="flex justify-between items-center text-[10px] font-mono text-gray-500 uppercase tracking-wider">
          <span>{message}</span>
          <span>{progress}%</span>
       </div>
    </div>
  );
}
