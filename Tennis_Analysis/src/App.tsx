import React, { useState } from 'react';
import { Activity, Database, Brain, Zap, BarChart3, Upload, Code } from 'lucide-react';
import Header from './components/Header';
import DatasetViewer from './components/DatasetViewer';
import MLPipeline from './components/MLPipeline';
import ModelDashboard from './components/ModelDashboard';
import PredictionInterface from './components/PredictionInterface';
import APIDemo from './components/APIDemo';

function App() {
  const [activeTab, setActiveTab] = useState('dataset');

  const tabs = [
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'pipeline', label: 'ML Pipeline', icon: Brain },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'predict', label: 'Predict', icon: Zap },
    { id: 'api', label: 'API Demo', icon: Code }
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'dataset':
        return <DatasetViewer />;
      case 'pipeline':
        return <MLPipeline />;
      case 'dashboard':
        return <ModelDashboard />;
      case 'predict':
        return <PredictionInterface />;
      case 'api':
        return <APIDemo />;
      default:
        return <DatasetViewer />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <Header />
      
      {/* Navigation Tabs */}
      <div className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-200 ${
                    activeTab === tab.id
                      ? 'border-green-500 text-green-600'
                      : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;