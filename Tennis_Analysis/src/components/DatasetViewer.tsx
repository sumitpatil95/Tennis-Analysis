import React, { useState } from 'react';
import { Eye, Download, FileText, Image, Target } from 'lucide-react';
import KeypointVisualization from './KeypointVisualization';
import DatasetStats from './DatasetStats';

const DatasetViewer: React.FC = () => {
  const [selectedAction, setSelectedAction] = useState('all');
  const [showKeypoints, setShowKeypoints] = useState(true);

  const actions = [
    { id: 'backhand', name: 'Backhand Shot', count: 500, color: 'bg-blue-500' },
    { id: 'forehand', name: 'Forehand Shot', count: 500, color: 'bg-green-500' },
    { id: 'ready_position', name: 'Ready Position', count: 500, color: 'bg-orange-500' },
    { id: 'serve', name: 'Serve', count: 500, color: 'bg-purple-500' }
  ];

  const keypointNames = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
  ];

  return (
    <div className="space-y-8">
      {/* Dataset Overview */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-slate-900">Dataset Overview</h2>
          <div className="flex items-center space-x-4">
            <button className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
              <Download className="w-4 h-4" />
              <span>Download Dataset</span>
            </button>
          </div>
        </div>

        <DatasetStats />
      </div>

      {/* Action Categories */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">Tennis Action Categories</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {actions.map((action) => (
            <div
              key={action.id}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                selectedAction === action.id
                  ? 'border-green-500 bg-green-50'
                  : 'border-slate-200 hover:border-slate-300'
              }`}
              onClick={() => setSelectedAction(action.id)}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-4 h-4 rounded-full ${action.color}`}></div>
                <div>
                  <h4 className="font-medium text-slate-900">{action.name}</h4>
                  <p className="text-sm text-slate-600">{action.count} images</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* COCO Annotations Info */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">COCO Format Annotations</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-slate-900 mb-3 flex items-center">
              <FileText className="w-5 h-5 mr-2 text-blue-500" />
              Annotation Structure
            </h4>
            <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm">
              <pre>{`{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [...],
      "num_keypoints": 18,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 1, "name": "backhand"},
    {"id": 2, "name": "forehand"},
    {"id": 3, "name": "ready_position"},
    {"id": 4, "name": "serve"}
  ]
}`}</pre>
            </div>
          </div>
          <div>
            <h4 className="font-medium text-slate-900 mb-3 flex items-center">
              <Target className="w-5 h-5 mr-2 text-green-500" />
              OpenPose Keypoints ({keypointNames.length} points)
            </h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {keypointNames.map((keypoint, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-slate-700">{keypoint}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Keypoint Visualization */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-slate-900">Keypoint Visualization</h3>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={showKeypoints}
              onChange={(e) => setShowKeypoints(e.target.checked)}
              className="rounded border-slate-300 text-green-500 focus:ring-green-500"
            />
            <span className="text-sm text-slate-700">Show Keypoints</span>
          </label>
        </div>
        <KeypointVisualization showKeypoints={showKeypoints} selectedAction={selectedAction} />
      </div>
    </div>
  );
};

export default DatasetViewer;