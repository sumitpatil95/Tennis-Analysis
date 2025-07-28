import React, { useState } from 'react';
import { Play, Pause, Settings, Download, Upload, BarChart3 } from 'lucide-react';
import TrainingProgress from './TrainingProgress';
import ModelConfiguration from './ModelConfiguration';

const MLPipeline: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  const pipelineSteps = [
    { id: 'preprocess', name: 'Data Preprocessing', status: 'completed' },
    { id: 'augment', name: 'Data Augmentation', status: 'completed' },
    { id: 'split', name: 'Train/Validation Split', status: 'completed' },
    { id: 'train', name: 'Model Training', status: 'running' },
    { id: 'evaluate', name: 'Model Evaluation', status: 'pending' },
    { id: 'deploy', name: 'Model Deployment', status: 'pending' }
  ];

  const handleStartTraining = () => {
    setIsTraining(true);
    // Simulate training progress
    setTimeout(() => setIsTraining(false), 10000);
  };

  return (
    <div className="space-y-8">
      {/* Pipeline Overview */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-slate-900">ML Pipeline</h2>
          <div className="flex items-center space-x-4">
            <button
              onClick={handleStartTraining}
              disabled={isTraining}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                isTraining
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-500 hover:bg-green-600'
              } text-white`}
            >
              {isTraining ? (
                <>
                  <Pause className="w-4 h-4" />
                  <span>Training...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Start Training</span>
                </>
              )}
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
              <Settings className="w-4 h-4" />
              <span>Configure</span>
            </button>
          </div>
        </div>

        {/* Pipeline Steps */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {pipelineSteps.map((step, index) => {
            const getStatusColor = (status: string) => {
              switch (status) {
                case 'completed': return 'bg-green-500';
                case 'running': return 'bg-blue-500 animate-pulse';
                case 'pending': return 'bg-gray-300';
                default: return 'bg-gray-300';
              }
            };

            return (
              <div key={step.id} className="text-center">
                <div className={`w-12 h-12 mx-auto rounded-full ${getStatusColor(step.status)} flex items-center justify-center text-white font-bold mb-2`}>
                  {index + 1}
                </div>
                <h3 className="font-medium text-sm text-slate-900">{step.name}</h3>
                <p className="text-xs text-slate-600 capitalize">{step.status}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Configuration */}
      <ModelConfiguration />

      {/* Training Progress */}
      {isTraining && <TrainingProgress />}

      {/* Data Processing */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">Data Processing Pipeline</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-slate-900 mb-3">Data Preprocessing Steps</h4>
            <ul className="space-y-2 text-sm text-slate-700">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Image resizing to 224x224 pixels</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Keypoint normalization</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>COCO annotation parsing</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Data validation and cleaning</span>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-slate-900 mb-3">Data Augmentation</h4>
            <ul className="space-y-2 text-sm text-slate-700">
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Random rotation (±15°)</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Horizontal flipping</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Color jittering</span>
              </li>
              <li className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Gaussian noise addition</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Model Architecture */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">Model Architecture</h3>
        <div className="bg-slate-50 rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h4 className="font-medium text-slate-900 mb-2">Input Layer</h4>
              <p className="text-sm text-slate-600">Images + Keypoints</p>
              <p className="text-xs text-slate-500">(224x224x3) + (18x2)</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h4 className="font-medium text-slate-900 mb-2">Feature Extraction</h4>
              <p className="text-sm text-slate-600">ResNet50 + MLP</p>
              <p className="text-xs text-slate-500">CNN + Dense Layers</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h4 className="font-medium text-slate-900 mb-2">Output Layer</h4>
              <p className="text-sm text-slate-600">4 Classes</p>
              <p className="text-xs text-slate-500">Softmax Activation</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLPipeline;