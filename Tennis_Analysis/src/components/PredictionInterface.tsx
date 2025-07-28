import React, { useState } from 'react';
import { Upload, Zap, Target, Image, Download } from 'lucide-react';

const PredictionInterface: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const sampleImages = [
    { url: 'https://images.pexels.com/photos/209977/pexels-photo-209977.jpeg?auto=compress&cs=tinysrgb&w=400', label: 'Sample Backhand' },
    { url: 'https://images.pexels.com/photos/1744411/pexels-photo-1744411.jpeg?auto=compress&cs=tinysrgb&w=400', label: 'Sample Forehand' },
    { url: 'https://images.pexels.com/photos/1103829/pexels-photo-1103829.jpeg?auto=compress&cs=tinysrgb&w=400', label: 'Sample Serve' },
    { url: 'https://images.pexels.com/photos/1752757/pexels-photo-1752757.jpeg?auto=compress&cs=tinysrgb&w=400', label: 'Sample Ready Position' }
  ];

  const handleImageSelect = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    setPrediction(null);
  };

  const handlePredict = () => {
    if (!selectedImage) return;
    
    setIsLoading(true);
    // Simulate prediction
    setTimeout(() => {
      const mockPrediction = {
        predicted_class: 'forehand',
        confidence: 0.94,
        probabilities: {
          backhand: 0.02,
          forehand: 0.94,
          ready_position: 0.03,
          serve: 0.01
        },
        keypoints_detected: 18,
        processing_time: 12
      };
      setPrediction(mockPrediction);
      setIsLoading(false);
    }, 2000);
  };

  return (
    <div className="space-y-8">
      {/* Upload Interface */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-slate-900 mb-6">Tennis Action Prediction</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Image Selection */}
          <div>
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Select or Upload Image</h3>
            
            {/* Upload Area */}
            <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors mb-4">
              <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
              <p className="text-slate-600 mb-2">Drop your tennis image here, or click to browse</p>
              <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                Choose File
              </button>
            </div>

            {/* Sample Images */}
            <div>
              <p className="text-sm font-medium text-slate-700 mb-3">Or choose a sample image:</p>
              <div className="grid grid-cols-2 gap-3">
                {sampleImages.map((image, index) => (
                  <div
                    key={index}
                    className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${
                      selectedImage === image.url
                        ? 'border-green-500 shadow-lg'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                    onClick={() => handleImageSelect(image.url)}
                  >
                    <img
                      src={image.url}
                      alt={image.label}
                      className="w-full h-24 object-cover"
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white text-xs p-1 text-center">
                      {image.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Selected Image Preview */}
          <div>
            <h3 className="text-lg font-semibold text-slate-900 mb-4">Image Preview</h3>
            {selectedImage ? (
              <div className="space-y-4">
                <div className="relative">
                  <img
                    src={selectedImage}
                    alt="Selected"
                    className="w-full rounded-lg shadow-md"
                  />
                  {/* Keypoint overlay would go here */}
                </div>
                <button
                  onClick={handlePredict}
                  disabled={isLoading}
                  className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-lg font-medium transition-colors ${
                    isLoading
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-green-500 hover:bg-green-600'
                  } text-white`}
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      <span>Predict Action</span>
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 bg-slate-100 rounded-lg">
                <div className="text-center">
                  <Image className="w-12 h-12 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-600">Select an image to preview</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-semibold text-slate-900 mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-green-500" />
            Prediction Results
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Main Prediction */}
            <div>
              <div className="bg-green-50 rounded-lg p-6 text-center mb-4">
                <h4 className="text-2xl font-bold text-green-900 mb-2">
                  {prediction.predicted_class.replace('_', ' ').toUpperCase()}
                </h4>
                <p className="text-lg text-green-700">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </p>
                <div className="mt-3">
                  <div className="bg-green-200 rounded-full h-3">
                    <div
                      className="bg-green-500 h-3 rounded-full transition-all duration-1000"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-slate-50 rounded-lg p-3 text-center">
                  <p className="font-medium text-slate-900">{prediction.keypoints_detected}</p>
                  <p className="text-slate-600">Keypoints Detected</p>
                </div>
                <div className="bg-slate-50 rounded-lg p-3 text-center">
                  <p className="font-medium text-slate-900">{prediction.processing_time}ms</p>
                  <p className="text-slate-600">Processing Time</p>
                </div>
              </div>
            </div>

            {/* All Probabilities */}
            <div>
              <h4 className="font-medium text-slate-900 mb-3">Class Probabilities</h4>
              <div className="space-y-3">
                {Object.entries(prediction.probabilities).map(([action, prob]: [string, any]) => (
                  <div key={action}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium text-slate-700">
                        {action.replace('_', ' ').toUpperCase()}
                      </span>
                      <span className="text-slate-600">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bg-slate-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-1000 ${
                          action === prediction.predicted_class
                            ? 'bg-green-500'
                            : 'bg-slate-400'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 flex space-x-2">
                <button className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors">
                  <Download className="w-4 h-4" />
                  <span>Export Results</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* API Integration */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">API Integration</h3>
        <div className="bg-slate-900 rounded-lg p-4 text-green-400 font-mono text-sm">
          <div className="mb-2 text-white"># Example API call</div>
          <div>curl -X POST https://api.tennisaction.ai/predict \</div>
          <div className="ml-4">-H "Content-Type: multipart/form-data" \</div>
          <div className="ml-4">-F "image=@tennis_action.jpg" \</div>
          <div className="ml-4">-H "Authorization: Bearer YOUR_API_KEY"</div>
        </div>
      </div>
    </div>
  );
};

export default PredictionInterface;