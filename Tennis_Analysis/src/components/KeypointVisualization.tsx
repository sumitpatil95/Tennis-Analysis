import React from 'react';
import { Eye, Download } from 'lucide-react';

interface KeypointVisualizationProps {
  showKeypoints: boolean;
  selectedAction: string;
}

const KeypointVisualization: React.FC<KeypointVisualizationProps> = ({ showKeypoints, selectedAction }) => {
  // Mock data for demonstration
  const sampleImages = [
    { id: 1, action: 'backhand', url: 'https://images.pexels.com/photos/209977/pexels-photo-209977.jpeg?auto=compress&cs=tinysrgb&w=400' },
    { id: 2, action: 'forehand', url: 'https://images.pexels.com/photos/1744411/pexels-photo-1744411.jpeg?auto=compress&cs=tinysrgb&w=400' },
    { id: 3, action: 'serve', url: 'https://images.pexels.com/photos/1103829/pexels-photo-1103829.jpeg?auto=compress&cs=tinysrgb&w=400' },
    { id: 4, action: 'ready_position', url: 'https://images.pexels.com/photos/1752757/pexels-photo-1752757.jpeg?auto=compress&cs=tinysrgb&w=400' }
  ];

  const keypoints = [
    { x: 50, y: 30, name: 'nose' },
    { x: 45, y: 25, name: 'left_eye' },
    { x: 55, y: 25, name: 'right_eye' },
    { x: 30, y: 45, name: 'left_shoulder' },
    { x: 70, y: 45, name: 'right_shoulder' },
    { x: 25, y: 60, name: 'left_elbow' },
    { x: 75, y: 60, name: 'right_elbow' },
    { x: 20, y: 75, name: 'left_wrist' },
    { x: 80, y: 75, name: 'right_wrist' }
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {sampleImages.map((image) => (
          <div key={image.id} className="relative group">
            <div className="relative overflow-hidden rounded-lg bg-slate-100">
              <img
                src={image.url}
                alt={`Tennis ${image.action}`}
                className="w-full h-48 object-cover transition-transform duration-300 group-hover:scale-105"
              />
              
              {/* Keypoint Overlay */}
              {showKeypoints && (
                <div className="absolute inset-0">
                  <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    {/* Skeleton connections */}
                    <line x1="50" y1="30" x2="30" y2="45" stroke="#22c55e" strokeWidth="0.5" />
                    <line x1="50" y1="30" x2="70" y2="45" stroke="#22c55e" strokeWidth="0.5" />
                    <line x1="30" y1="45" x2="25" y2="60" stroke="#22c55e" strokeWidth="0.5" />
                    <line x1="70" y1="45" x2="75" y2="60" stroke="#22c55e" strokeWidth="0.5" />
                    <line x1="25" y1="60" x2="20" y2="75" stroke="#22c55e" strokeWidth="0.5" />
                    <line x1="75" y1="60" x2="80" y2="75" stroke="#22c55e" strokeWidth="0.5" />
                    
                    {/* Keypoints */}
                    {keypoints.map((point, index) => (
                      <circle
                        key={index}
                        cx={point.x}
                        cy={point.y}
                        r="1.5"
                        fill="#22c55e"
                        stroke="white"
                        strokeWidth="0.5"
                        className="opacity-80"
                      />
                    ))}
                  </svg>
                </div>
              )}
              
              {/* Action Label */}
              <div className="absolute bottom-2 left-2 px-2 py-1 bg-black bg-opacity-75 text-white text-xs rounded">
                {image.action.replace('_', ' ').toUpperCase()}
              </div>
              
              {/* Hover Actions */}
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
                <div className="flex space-x-2">
                  <button className="p-2 bg-white bg-opacity-90 rounded-full hover:bg-opacity-100 transition-all">
                    <Eye className="w-4 h-4 text-slate-700" />
                  </button>
                  <button className="p-2 bg-white bg-opacity-90 rounded-full hover:bg-opacity-100 transition-all">
                    <Download className="w-4 h-4 text-slate-700" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="text-center">
        <button className="px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
          Load More Images
        </button>
      </div>
    </div>
  );
};

export default KeypointVisualization;