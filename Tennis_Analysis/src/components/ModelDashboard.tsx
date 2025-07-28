import React from 'react';
import { Award, TrendingUp, Clock, Download, Play } from 'lucide-react';
import PerformanceMetrics from './PerformanceMetrics';
import ConfusionMatrix from './ConfusionMatrix';

const ModelDashboard: React.FC = () => {
  const modelStats = [
    {
      label: 'Best Accuracy',
      value: '94.2%',
      icon: Award,
      color: 'text-green-500',
      bgColor: 'bg-green-50'
    },
    {
      label: 'Training Time',
      value: '2h 35m',
      icon: Clock,
      color: 'text-blue-500',
      bgColor: 'bg-blue-50'
    },
    {
      label: 'Model Size',
      value: '45.2 MB',
      icon: Download,
      color: 'text-purple-500',
      bgColor: 'bg-purple-50'
    },
    {
      label: 'Inference Speed',
      value: '12ms',
      icon: TrendingUp,
      color: 'text-orange-500',
      bgColor: 'bg-orange-50'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Model Overview */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-slate-900">Model Dashboard</h2>
          <div className="flex items-center space-x-4">
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              Training Complete
            </span>
            <button className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
              <Play className="w-4 h-4" />
              <span>Deploy Model</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {modelStats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div key={index} className={`${stat.bgColor} rounded-lg p-4`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-600">{stat.label}</p>
                    <p className="text-2xl font-bold text-slate-900">{stat.value}</p>
                  </div>
                  <Icon className={`w-8 h-8 ${stat.color}`} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Performance Metrics */}
      <PerformanceMetrics />

      {/* Confusion Matrix */}
      <ConfusionMatrix />

      {/* Model Information */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">Model Information</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-slate-900 mb-3">Architecture Details</h4>
            <ul className="space-y-2 text-sm text-slate-700">
              <li><strong>Model Type:</strong> CNN + Keypoint MLP</li>
              <li><strong>Backbone:</strong> ResNet50</li>
              <li><strong>Input Shape:</strong> (224, 224, 3) + (18, 2)</li>
              <li><strong>Output Classes:</strong> 4 (backhand, forehand, ready_position, serve)</li>
              <li><strong>Total Parameters:</strong> 25.6M</li>
              <li><strong>Trainable Parameters:</strong> 23.5M</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-slate-900 mb-3">Training Configuration</h4>
            <ul className="space-y-2 text-sm text-slate-700">
              <li><strong>Optimizer:</strong> Adam (lr=0.001)</li>
              <li><strong>Loss Function:</strong> Categorical Crossentropy</li>
              <li><strong>Batch Size:</strong> 32</li>
              <li><strong>Epochs:</strong> 100</li>
              <li><strong>Early Stopping:</strong> Patience 10</li>
              <li><strong>Data Split:</strong> 80% Train, 20% Validation</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Export Options */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">Export & Deployment</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-4 border-2 border-dashed border-slate-300 rounded-lg hover:border-blue-400 transition-colors text-center">
            <Download className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <p className="font-medium text-slate-900">TensorFlow SavedModel</p>
            <p className="text-sm text-slate-600">For production deployment</p>
          </button>
          <button className="p-4 border-2 border-dashed border-slate-300 rounded-lg hover:border-blue-400 transition-colors text-center">
            <Download className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <p className="font-medium text-slate-900">ONNX Format</p>
            <p className="text-sm text-slate-600">Cross-platform inference</p>
          </button>
          <button className="p-4 border-2 border-dashed border-slate-300 rounded-lg hover:border-blue-400 transition-colors text-center">
            <Download className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <p className="font-medium text-slate-900">TensorFlow Lite</p>
            <p className="text-sm text-slate-600">Mobile deployment</p>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelDashboard;