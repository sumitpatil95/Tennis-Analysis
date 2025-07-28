import React from 'react';
import { Image, Target, Layers, Database } from 'lucide-react';

const DatasetStats: React.FC = () => {
  const stats = [
    {
      label: 'Total Images',
      value: '2,000',
      icon: Image,
      color: 'text-blue-500',
      bgColor: 'bg-blue-50'
    },
    {
      label: 'Action Categories',
      value: '4',
      icon: Layers,
      color: 'text-green-500',
      bgColor: 'bg-green-50'
    },
    {
      label: 'Keypoints per Image',
      value: '18',
      icon: Target,
      color: 'text-orange-500',
      bgColor: 'bg-orange-50'
    },
    {
      label: 'Annotation Format',
      value: 'COCO',
      icon: Database,
      color: 'text-purple-500',
      bgColor: 'bg-purple-50'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => {
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
  );
};

export default DatasetStats;