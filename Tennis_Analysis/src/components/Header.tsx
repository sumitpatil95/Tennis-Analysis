import React from 'react';
import { Activity } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-lg border-b-4 border-green-500">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-500 rounded-lg">
              <Activity className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Tennis Action Recognition</h1>
              <p className="text-sm text-slate-600">AI/ML Pipeline for Tennis Action Analysis</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-slate-600">
              <span className="font-medium">Dataset:</span> 2,000 images | 4 actions
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;