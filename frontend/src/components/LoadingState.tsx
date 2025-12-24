import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Newspaper, BrainCircuit, LineChart } from 'lucide-react';

const steps = [
  { icon: Newspaper, label: "Fetching latest macro news..." },
  { icon: BrainCircuit, label: "Training micro-model on recent data..." },
  { icon: LineChart, label: "Generating investment analysis..." },
];

export function LoadingState() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    // Simulate step progression matching
    const timer1 = setTimeout(() => setActiveStep(1), 8000);  // News
    const timer2 = setTimeout(() => setActiveStep(2), 25000); // Training
    
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-8 animate-in fade-in zoom-in duration-500">
      <div className="relative">
        <div className="absolute inset-0 bg-blue-500/20 blur-xl rounded-full" />
        <Loader2 className="w-16 h-16 text-blue-500 animate-spin relative z-10" />
      </div>

      <div className="space-y-4 w-full max-w-sm">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          const isActive = idx === activeStep;
          const isCompleted = idx < activeStep;

          return (
            <div 
              key={idx}
              className={`flex items-center space-x-3 transition-all duration-500 ${
                isActive ? 'opacity-100 scale-105' : 
                isCompleted ? 'opacity-50' : 'opacity-30'
              }`}
            >
              <div className={`p-2 rounded-full ${
                isActive ? 'bg-blue-500/10 text-blue-500' : 
                isCompleted ? 'bg-green-500/10 text-green-500' : 'bg-gray-800 text-gray-500'
              }`}>
                <Icon className="w-5 h-5" />
              </div>
              <span className={`text-sm font-medium ${
                isActive ? 'text-blue-400' : 
                isCompleted ? 'text-green-400' : 'text-gray-500'
              }`}>
                {step.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
