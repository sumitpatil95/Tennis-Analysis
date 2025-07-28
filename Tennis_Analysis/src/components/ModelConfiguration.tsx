import React, { useState } from 'react';
import { Settings, Save } from 'lucide-react';

const ModelConfiguration: React.FC = () => {
  const [config, setConfig] = useState({
    batchSize: 32,
    learningRate: 0.001,
    epochs: 100,
    optimizer: 'adam',
    lossFunction: 'categorical_crossentropy',
    validationSplit: 0.2,
    earlyStoppingPatience: 10,
    reduceOnPlateau: true
  });

  const handleConfigChange = (key: string, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-slate-900 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-blue-500" />
          Model Configuration
        </h3>
        <button className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
          <Save className="w-4 h-4" />
          <span>Save Config</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Batch Size
          </label>
          <input
            type="number"
            value={config.batchSize}
            onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Learning Rate
          </label>
          <input
            type="number"
            step="0.0001"
            value={config.learningRate}
            onChange={(e) => handleConfigChange('learningRate', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Epochs
          </label>
          <input
            type="number"
            value={config.epochs}
            onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Optimizer
          </label>
          <select
            value={config.optimizer}
            onChange={(e) => handleConfigChange('optimizer', e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
            <option value="rmsprop">RMSprop</option>
            <option value="adamw">AdamW</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Loss Function
          </label>
          <select
            value={config.lossFunction}
            onChange={(e) => handleConfigChange('lossFunction', e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="categorical_crossentropy">Categorical Crossentropy</option>
            <option value="sparse_categorical_crossentropy">Sparse Categorical Crossentropy</option>
            <option value="focal_loss">Focal Loss</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Validation Split
          </label>
          <input
            type="number"
            step="0.1"
            min="0.1"
            max="0.5"
            value={config.validationSplit}
            onChange={(e) => handleConfigChange('validationSplit', parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      <div className="mt-6 flex items-center space-x-6">
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={config.reduceOnPlateau}
            onChange={(e) => handleConfigChange('reduceOnPlateau', e.target.checked)}
            className="rounded border-slate-300 text-blue-500 focus:ring-blue-500"
          />
          <span className="text-sm text-slate-700">Reduce LR on Plateau</span>
        </label>
        
        <div className="flex items-center space-x-2">
          <label className="text-sm text-slate-700">Early Stopping Patience:</label>
          <input
            type="number"
            value={config.earlyStoppingPatience}
            onChange={(e) => handleConfigChange('earlyStoppingPatience', parseInt(e.target.value))}
            className="w-20 px-2 py-1 border border-slate-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>
    </div>
  );
};

export default ModelConfiguration;