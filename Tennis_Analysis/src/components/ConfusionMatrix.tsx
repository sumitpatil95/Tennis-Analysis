import React from 'react';
import { Grid3X3 } from 'lucide-react';

const ConfusionMatrix: React.FC = () => {
  const classes = ['Backhand', 'Forehand', 'Ready Position', 'Serve'];
  const matrix = [
    [92, 3, 4, 1],
    [2, 95, 2, 1],
    [5, 2, 93, 0],
    [1, 0, 3, 96]
  ];

  const getIntensity = (value: number, max: number) => {
    const intensity = value / max;
    if (intensity > 0.8) return 'bg-green-600 text-white';
    if (intensity > 0.6) return 'bg-green-500 text-white';
    if (intensity > 0.4) return 'bg-green-400 text-white';
    if (intensity > 0.2) return 'bg-green-300 text-slate-900';
    if (intensity > 0) return 'bg-green-100 text-slate-900';
    return 'bg-slate-50 text-slate-600';
  };

  const maxValue = Math.max(...matrix.flat());

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-semibold text-slate-900 mb-4 flex items-center">
        <Grid3X3 className="w-5 h-5 mr-2 text-purple-500" />
        Confusion Matrix
      </h3>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="grid grid-cols-5 gap-1 text-sm">
            {/* Header row */}
            <div className="p-2"></div>
            {classes.map((cls, index) => (
              <div key={index} className="p-2 text-center font-medium text-slate-700 bg-slate-100 rounded">
                {cls}
              </div>
            ))}

            {/* Matrix rows */}
            {matrix.map((row, rowIndex) => (
              <React.Fragment key={rowIndex}>
                <div className="p-2 font-medium text-slate-700 bg-slate-100 rounded flex items-center justify-center">
                  {classes[rowIndex]}
                </div>
                {row.map((value, colIndex) => (
                  <div
                    key={colIndex}
                    className={`p-3 text-center font-medium rounded transition-colors ${getIntensity(value, maxValue)}`}
                  >
                    {value}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-slate-600">
        <span>Predicted Label</span>
        <div className="flex items-center space-x-4">
          <span>True Label</span>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-100 rounded"></div>
            <span>Low</span>
            <div className="w-3 h-3 bg-green-600 rounded"></div>
            <span>High</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfusionMatrix;