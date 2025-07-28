import React, { useState } from 'react';
import { Upload, Zap, Code, Play, CheckCircle, AlertCircle } from 'lucide-react';
import axios from 'axios';

const APIDemo: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async (modelType: 'classical' | 'deep' | 'ensemble') => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      if (modelType === 'classical') {
        formData.append('model_name', 'random_forest');
      } else if (modelType === 'deep') {
        formData.append('model_name', 'hybrid_cnn');
      }

      const response = await axios.post(`${apiUrl}/predict/${modelType}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const codeExamples = {
    python: `import requests

# Predict tennis action
files = {'file': open('tennis_image.jpg', 'rb')}
response = requests.post('${apiUrl}/predict/ensemble', files=files)
result = response.json()

print(f"Action: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")`,
    
    curl: `curl -X POST "${apiUrl}/predict/ensemble" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@tennis_image.jpg"`,
    
    javascript: `const formData = new FormData();
formData.append('file', imageFile);

fetch('${apiUrl}/predict/ensemble', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Action:', data.predicted_class);
  console.log('Confidence:', data.confidence);
});`
  };

  return (
    <div className="space-y-8">
      {/* API Configuration */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-slate-900 mb-4">API Demo & Testing</h2>
        
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            API Base URL
          </label>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
            placeholder="http://localhost:8000"
          />
        </div>

        {/* File Upload */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Upload Tennis Image
          </label>
          <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-green-400 transition-colors">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
              <p className="text-slate-600 mb-2">
                {selectedFile ? selectedFile.name : 'Click to upload or drag and drop'}
              </p>
              <p className="text-sm text-slate-500">PNG, JPG up to 10MB</p>
            </label>
          </div>
        </div>

        {/* Prediction Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <button
            onClick={() => handlePredict('classical')}
            disabled={!selectedFile || isLoading}
            className="flex items-center justify-center space-x-2 px-4 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            <Code className="w-5 h-5" />
            <span>Classical ML</span>
          </button>
          
          <button
            onClick={() => handlePredict('deep')}
            disabled={!selectedFile || isLoading}
            className="flex items-center justify-center space-x-2 px-4 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            <Zap className="w-5 h-5" />
            <span>Deep Learning</span>
          </button>
          
          <button
            onClick={() => handlePredict('ensemble')}
            disabled={!selectedFile || isLoading}
            className="flex items-center justify-center space-x-2 px-4 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            <Play className="w-5 h-5" />
            <span>Ensemble</span>
          </button>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500 mx-auto mb-2"></div>
            <p className="text-slate-600">Analyzing tennis action...</p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <CheckCircle className="w-6 h-6 text-green-500 mr-2" />
              <h3 className="text-lg font-semibold text-green-900">Prediction Results</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Main Prediction</h4>
                <div className="bg-white rounded-lg p-4">
                  <p className="text-2xl font-bold text-green-600 mb-1">
                    {prediction.predicted_class.replace('_', ' ').toUpperCase()}
                  </p>
                  <p className="text-slate-600">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-slate-500 mt-2">
                    Model: {prediction.model_name || prediction.model_type}
                  </p>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-slate-900 mb-2">All Probabilities</h4>
                <div className="space-y-2">
                  {Object.entries(prediction.probabilities || {}).map(([action, prob]: [string, any]) => (
                    <div key={action} className="flex items-center justify-between">
                      <span className="text-sm font-medium text-slate-700">
                        {action.replace('_', ' ').toUpperCase()}
                      </span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-slate-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-slate-600 w-12">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {prediction.individual_predictions && (
              <div className="mt-6">
                <h4 className="font-medium text-slate-900 mb-2">Individual Model Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(prediction.individual_predictions).map(([modelName, pred]: [string, any]) => (
                    <div key={modelName} className="bg-white rounded-lg p-3">
                      <p className="font-medium text-slate-800">{modelName.replace('_', ' ').title()}</p>
                      <p className="text-sm text-slate-600">
                        {pred.predicted_class} ({(pred.confidence * 100).toFixed(1)}%)
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Code Examples */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">API Usage Examples</h3>
        
        <div className="space-y-6">
          {Object.entries(codeExamples).map(([language, code]) => (
            <div key={language}>
              <h4 className="font-medium text-slate-900 mb-2 capitalize">{language}</h4>
              <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-green-400 text-sm">
                  <code>{code}</code>
                </pre>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* API Documentation */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-slate-900 mb-4">API Endpoints</h3>
        
        <div className="space-y-4">
          <div className="border border-slate-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm font-medium">GET</span>
              <code className="text-slate-700">/health</code>
            </div>
            <p className="text-slate-600 text-sm">Check API health and loaded models</p>
          </div>
          
          <div className="border border-slate-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm font-medium">POST</span>
              <code className="text-slate-700">/predict/classical</code>
            </div>
            <p className="text-slate-600 text-sm">Predict using classical ML models (Random Forest, SVM, etc.)</p>
          </div>
          
          <div className="border border-slate-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-sm font-medium">POST</span>
              <code className="text-slate-700">/predict/deep</code>
            </div>
            <p className="text-slate-600 text-sm">Predict using deep learning models (MLP, CNN)</p>
          </div>
          
          <div className="border border-slate-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <span className="px-2 py-1 bg-orange-100 text-orange-800 rounded text-sm font-medium">POST</span>
              <code className="text-slate-700">/predict/ensemble</code>
            </div>
            <p className="text-slate-600 text-sm">Predict using ensemble of all available models</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIDemo;