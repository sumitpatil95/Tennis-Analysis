import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Clock, Target } from 'lucide-react';

const TrainingProgress: React.FC = () => {
  const [progress, setProgress] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    loss: 1.0,
    valAccuracy: 0,
    valLoss: 1.0
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => Math.min(prev + 2, 100));
      setEpoch(prev => Math.min(prev + 1, 100));
      setMetrics(prev => ({
        accuracy: Math.min(prev.accuracy + 0.02, 0.95),
        loss: Math.max(prev.loss - 0.02, 0.05),
        valAccuracy: Math.min(prev.valAccuracy + 0.015, 0.92),
        valLoss: Math.max(prev.valLoss - 0.015, 0.08)
      }));
    }, 500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-semibold text-slate-900 mb-4">Training Progress</h3>
      
      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-slate-700 mb-2">
          <span>Epoch {epoch}/100</span>
          <span>{progress}% Complete</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-3">
          <div 
            className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-700">Training Accuracy</p>
              <p className="text-2xl font-bold text-green-900">{(metrics.accuracy * 100).toFixed(1)}%</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-red-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-red-700">Training Loss</p>
              <p className="text-2xl font-bold text-red-900">{metrics.loss.toFixed(3)}</p>
            </div>
            <TrendingDown className="w-8 h-8 text-red-500" />
          </div>
        </div>

        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-700">Validation Accuracy</p>
              <p className="text-2xl font-bold text-blue-900">{(metrics.valAccuracy * 100).toFixed(1)}%</p>
            </div>
            <Target className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-orange-50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-orange-700">Validation Loss</p>
              <p className="text-2xl font-bold text-orange-900">{metrics.valLoss.toFixed(3)}</p>
            </div>
            <Clock className="w-8 h-8 text-orange-500" />
          </div>
        </div>
      </div>

      {/* Training Logs */}
      <div className="bg-slate-900 rounded-lg p-4 text-green-400 font-mono text-sm max-h-40 overflow-y-auto">
        <div>Epoch {epoch}/100</div>
        <div>1600/1600 [==============================] - 23s 14ms/step</div>
        <div>loss: {metrics.loss.toFixed(4)} - accuracy: {metrics.accuracy.toFixed(4)} - val_loss: {metrics.valLoss.toFixed(4)} - val_accuracy: {metrics.valAccuracy.toFixed(4)}</div>
        <div>Learning rate: 0.001</div>
        <div className="text-yellow-400">Info: Training is progressing normally...</div>
      </div>
    </div>
  );
};

export default TrainingProgress;