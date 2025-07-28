import React from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';

const PerformanceMetrics: React.FC = () => {
  const classMetrics = [
    { class: 'Backhand', precision: 0.94, recall: 0.92, f1: 0.93, support: 100 },
    { class: 'Forehand', precision: 0.96, recall: 0.95, f1: 0.95, support: 100 },
    { class: 'Ready Position', precision: 0.91, recall: 0.93, f1: 0.92, support: 100 },
    { class: 'Serve', precision: 0.97, recall: 0.96, f1: 0.96, support: 100 }
  ];

  const overallMetrics = {
    accuracy: 0.945,
    macroAvg: { precision: 0.945, recall: 0.94, f1: 0.94 },
    weightedAvg: { precision: 0.945, recall: 0.945, f1: 0.945 }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-xl font-semibold text-slate-900 mb-4 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2 text-blue-500" />
        Performance Metrics
      </h3>

      {/* Class-wise Metrics */}
      <div className="mb-6">
        <h4 className="font-medium text-slate-900 mb-3">Class-wise Performance</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-4 py-2 text-left font-medium text-slate-700">Class</th>
                <th className="px-4 py-2 text-center font-medium text-slate-700">Precision</th>
                <th className="px-4 py-2 text-center font-medium text-slate-700">Recall</th>
                <th className="px-4 py-2 text-center font-medium text-slate-700">F1-Score</th>
                <th className="px-4 py-2 text-center font-medium text-slate-700">Support</th>
              </tr>
            </thead>
            <tbody>
              {classMetrics.map((metric, index) => (
                <tr key={index} className="border-t border-slate-200">
                  <td className="px-4 py-3 font-medium text-slate-900">{metric.class}</td>
                  <td className="px-4 py-3 text-center">
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-medium">
                      {(metric.precision * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
                      {(metric.recall * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs font-medium">
                      {(metric.f1 * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center text-slate-600">{metric.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Overall Metrics */}
      <div className="bg-slate-50 rounded-lg p-4">
        <h4 className="font-medium text-slate-900 mb-3">Overall Performance</h4>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">{(overallMetrics.accuracy * 100).toFixed(1)}%</p>
            <p className="text-sm text-slate-600">Accuracy</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{(overallMetrics.macroAvg.precision * 100).toFixed(1)}%</p>
            <p className="text-sm text-slate-600">Macro Avg Precision</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">{(overallMetrics.macroAvg.recall * 100).toFixed(1)}%</p>
            <p className="text-sm text-slate-600">Macro Avg Recall</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">{(overallMetrics.macroAvg.f1 * 100).toFixed(1)}%</p>
            <p className="text-sm text-slate-600">Macro Avg F1</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetrics;