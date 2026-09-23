import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
});

export interface AnalysisResult {
  classification: 'SAFE' | 'SUSPICIOUS' | 'PHISHING';
  risk_score: number;
  ml_probability: number;
  indicators: string[];
  recommendations: string[];
  explanation: string;
  score_breakdown: {
    ml_component: number;
    url_component: number;
    content_component: number;
    behavioral_component: number;
  };
  input_type: string;
  scan_id: number;
}

export interface HistoryItem {
  id: number;
  input_type: string;
  input_text: string;
  classification: string;
  risk_score: number;
  ml_probability: number;
  indicators: string[];
  recommendations: string[];
  explanation: string;
  timestamp: string;
}

export interface Statistics {
  total_scans: number;
  phishing_count: number;
  suspicious_count: number;
  safe_count: number;
  average_risk_score: number;
  recent_scans: Array<{
    classification: string;
    risk_score: number;
    timestamp: string;
  }>;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  ml_models: {
    url_model_loaded: boolean;
    message_model_loaded: boolean;
  };
}

export const checkHealth = async (): Promise<HealthStatus> => {
  const response = await api.get('/health');
  return response.data;
};

export const analyzeUrl = async (url: string): Promise<AnalysisResult> => {
  const response = await api.post('/analyze/url', { url });
  return response.data;
};

export const analyzeMessage = async (message: string): Promise<AnalysisResult> => {
  const response = await api.post('/analyze/message', { message });
  return response.data;
};

export const getHistory = async (limit = 50): Promise<{ history: HistoryItem[]; count: number }> => {
  const response = await api.get(`/history?limit=${limit}`);
  return response.data;
};

export const getStatistics = async (): Promise<Statistics> => {
  const response = await api.get('/statistics');
  return response.data;
};

export default api;
