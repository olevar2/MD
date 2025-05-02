import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Brush
} from 'recharts';
import { Card, Select, Button, Tabs, Table, Slider, Spin, Alert, Tag, Space, Row, Col, Statistic } from 'antd';

import { getTrainingHistory, getModelPerformance, getModelFeatureImportance } from '../../api/ml-workbench';

const { TabPane } = Tabs;
const { Option } = Select;

/**
 * Component for visualizing RL model training progress and performance metrics
 */
const RLTrainingVisualization = ({ modelId, comparisonModels = [] }) => {
  const [activeTab, setActiveTab] = useState('1');
  const [trainingData, setTrainingData] = useState([]);
  const [performanceData, setPerformanceData] = useState({});
  const [featureImportance, setFeatureImportance] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('all');
  const [metricType, setMetricType] = useState('reward');

  useEffect(() => {
    if (modelId) {
      fetchTrainingData();
      fetchPerformanceData();
      fetchFeatureImportance();
    }
  }, [modelId, timeRange]);

  const fetchTrainingData = async () => {
    try {
      setIsLoading(true);
      const data = await getTrainingHistory(modelId, timeRange);
      setTrainingData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load training data');
      console.error('Error fetching training data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchPerformanceData = async () => {
    try {
      setIsLoading(true);
      const data = await getModelPerformance(modelId);
      setPerformanceData(data);
      setError(null);
    } catch (err) {
      setError('Failed to load performance data');
      console.error('Error fetching performance data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchFeatureImportance = async () => {
    try {
      setIsLoading(true);
      const data = await getModelFeatureImportance(modelId);
      setFeatureImportance(data);
      setError(null);
    } catch (err) {
      setError('Failed to load feature importance data');
      console.error('Error fetching feature importance:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Format colors for consistent visualization
  const colors = {
    reward: '#8884d8',
    pnl: '#82ca9d',
    sharpe: '#ff7300',
    drawdown: '#ff0000',
    volatility: '#8dd1e1',
    winRate: '#a4de6c',
    comparison: ['#0088FE', '#00C49F', '#FFBB28', '#FF8042']
  };

  return (
    <div className="rl-training-visualization">
      {error && <Alert message={error} type="error" showIcon style={{ marginBottom: 16 }} />}
      
      <Card
        title="RL Training Progress"
        extra={
          <Space>
            <Select defaultValue={timeRange} onChange={setTimeRange}>
              <Option value="1h">Last Hour</Option>
              <Option value="6h">Last 6 Hours</Option>
              <Option value="24h">Last 24 Hours</Option>
              <Option value="7d">Last 7 Days</Option>
              <Option value="all">All Time</Option>
            </Select>
            <Button type="primary" onClick={() => { fetchTrainingData(); fetchPerformanceData(); }}>
              Refresh
            </Button>
          </Space>
        }
      >
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="Training Progress" key="1">
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card title="Reward Evolution" size="small">
                  <ResponsiveContainer width="100%" height={300}>
                    {isLoading ? <Spin /> : (
                      <LineChart data={trainingData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="episode" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="reward" stroke={colors.reward} dot={false} />
                        {comparisonModels.map((model, index) => (
                          <Line 
                            key={model.id}
                            type="monotone"
                            dataKey="reward"
                            data={model.trainingData || []}
                            stroke={colors.comparison[index % colors.comparison.length]}
                            dot={false}
                            name={`${model.name} Reward`}
                          />
                        ))}
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </Card>
              </Col>
            </Row>
            
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Card title="Trading Performance" size="small">
                  <ResponsiveContainer width="100%" height={250}>
                    {isLoading ? <Spin /> : (
                      <LineChart data={trainingData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="episode" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="pnl" stroke={colors.pnl} dot={false} />
                        <Line type="monotone" dataKey="sharpe" stroke={colors.sharpe} dot={false} />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="Risk Metrics" size="small">
                  <ResponsiveContainer width="100%" height={250}>
                    {isLoading ? <Spin /> : (
                      <LineChart data={trainingData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="episode" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="drawdown" stroke={colors.drawdown} dot={false} />
                        <Line type="monotone" dataKey="volatility" stroke={colors.volatility} dot={false} />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </Card>
              </Col>
            </Row>
          </TabPane>
          
          <TabPane tab="Performance Metrics" key="2">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Sharpe Ratio"
                    value={performanceData.sharpe_ratio}
                    precision={2}
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Max Drawdown"
                    value={performanceData.max_drawdown}
                    precision={2}
                    valueStyle={{ color: '#cf1322' }}
                    suffix="%"
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Win Rate"
                    value={performanceData.win_rate}
                    precision={2}
                    valueStyle={{ color: '#3f8600' }}
                    suffix="%"
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Total Return"
                    value={performanceData.total_return}
                    precision={2}
                    valueStyle={{ color: performanceData.total_return >= 0 ? '#3f8600' : '#cf1322' }}
                    suffix="%"
                  />
                </Card>
              </Col>
            </Row>
            
            <Card title="Regime Performance" style={{ marginTop: 16 }}>
              <ResponsiveContainer width="100%" height={300}>
                {isLoading ? <Spin /> : (
                  <BarChart data={performanceData.regime_performance || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="regime" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="return" fill="#8884d8" name="Return %" />
                    <Bar dataKey="sharpe" fill="#82ca9d" name="Sharpe Ratio" />
                    <Bar dataKey="trades" fill="#ffc658" name="# Trades" />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </Card>
          </TabPane>
          
          <TabPane tab="Model Explainability" key="3">
            <Card title="Feature Importance">
              <ResponsiveContainer width="100%" height={400}>
                {isLoading ? <Spin /> : (
                  <BarChart
                    data={featureImportance}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" width={150} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="importance" fill="#8884d8" />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </Card>
            
            <Card title="Critical State Explorer" style={{ marginTop: 16 }}>
              <Alert
                message="State Importance Analysis"
                description="This visualization shows states that had the highest impact on model decisions. Explore critical trading moments to understand the model's behavior."
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              <ResponsiveContainer width="100%" height={400}>
                {isLoading ? <Spin /> : (
                  <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                  >
                    <CartesianGrid />
                    <XAxis type="number" dataKey="actionValue" name="Action Value" />
                    <YAxis type="number" dataKey="stateCriticality" name="State Criticality" />
                    <ZAxis type="number" dataKey="reward" range={[50, 400]} name="Reward" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Legend />
                    <Scatter
                      name="Critical States"
                      data={performanceData.critical_states || []}
                      fill="#8884d8"
                    />
                  </ScatterChart>
                )}
              </ResponsiveContainer>
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default RLTrainingVisualization;
