import React, { useState, useEffect } from 'react';
import { Tabs, Card, Divider, Button, Select, Tooltip, Spin, Table, Tag } from 'antd';
import { QuestionCircleOutlined, FileSearchOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

// This would be a proper chart library in production
// For this prototype, we're simulating visualizations
const Heatmap = ({ data }: { data: any }) => (
  <div className="border p-4 rounded bg-gray-50">
    <div className="text-center font-medium mb-2">State-Action Value Heatmap</div>
    <div className="grid grid-cols-5 gap-1">
      {Array(25).fill(0).map((_, i) => {
        const intensity = Math.random(); // Would be actual values from the model
        const color = `rgba(0, 128, 255, ${intensity})`;
        return (
          <div 
            key={i} 
            style={{
              backgroundColor: color,
              height: '30px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: intensity > 0.5 ? 'white' : 'black',
              fontSize: '12px'
            }}
          >
            {intensity.toFixed(2)}
          </div>
        );
      })}
    </div>
    <div className="text-center text-xs mt-2 text-gray-500">*Visualization shows state-action values across the feature space</div>
  </div>
);

const AttentionVisualization = ({ data }: { data: any }) => (
  <div className="border p-4 rounded bg-gray-50">
    <div className="text-center font-medium mb-2">Attention Mechanism Visualization</div>
    <div className="relative h-40">
      {/* This would be an actual attention visualization with D3 or a specialized chart library */}
      <div className="flex justify-between">
        {Array(8).fill(0).map((_, i) => (
          <div key={i} className="flex flex-col items-center">
            <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs">
              {i+1}
            </div>
            {Array(3).fill(0).map((_, j) => {
              const opacity = Math.random();
              return (
                <div 
                  key={`${i}-${j}`} 
                  className="mt-1 w-4 h-4 rounded-full"
                  style={{ backgroundColor: `rgba(255, 99, 71, ${opacity})` }}
                ></div>
              );
            })}
          </div>
        ))}
      </div>
      {/* Connection lines would be drawn here with SVG */}
    </div>
    <div className="text-center text-xs mt-2 text-gray-500">*Visualization shows which inputs the model is focusing on</div>
  </div>
);

interface FeatureImportance {
  feature: string;
  importance: number;
  category: string;
}

interface StateAction {
  state: string;
  action: string;
  value: number;
  confidence: number;
  explanation: string;
}

interface CriticalState {
  id: string;
  timestamp: string;
  description: string;
  marketRegime: string;
  actionTaken: string;
  outcome: string;
  rewardValue: number;
}

const { TabPane } = Tabs;
const { Option } = Select;

// Mock data for the feature importance
const mockFeatureImportance: FeatureImportance[] = [
  { feature: 'close_price_1m', importance: 0.85, category: 'price' },
  { feature: 'volume_1m', importance: 0.72, category: 'volume' },
  { feature: 'ma_crossover_5m', importance: 0.68, category: 'indicator' },
  { feature: 'rsi_1m', importance: 0.64, category: 'indicator' },
  { feature: 'spread_1m', importance: 0.61, category: 'market' },
  { feature: 'volatility_5m', importance: 0.58, category: 'market' },
  { feature: 'order_book_imbalance', importance: 0.55, category: 'orderbook' },
  { feature: 'position_duration', importance: 0.51, category: 'position' },
  { feature: 'position_size', importance: 0.48, category: 'position' },
  { feature: 'unrealized_pnl', importance: 0.45, category: 'position' },
  { feature: 'market_regime', importance: 0.42, category: 'market' },
  { feature: 'time_of_day', importance: 0.38, category: 'time' },
  { feature: 'day_of_week', importance: 0.35, category: 'time' },
  { feature: 'bollinger_width_15m', importance: 0.32, category: 'indicator' },
  { feature: 'support_level_distance', importance: 0.28, category: 'indicator' },
];

// Mock data for critical states
const mockCriticalStates: CriticalState[] = [
  {
    id: 'cs-001',
    timestamp: '2025-04-14 09:32:15',
    description: 'High volatility near major support level',
    marketRegime: 'Volatile',
    actionTaken: 'Hold',
    outcome: 'Positive',
    rewardValue: 0.85,
  },
  {
    id: 'cs-002',
    timestamp: '2025-04-14 10:45:22',
    description: 'Failed breakout with high volume',
    marketRegime: 'Ranging',
    actionTaken: 'Sell',
    outcome: 'Negative',
    rewardValue: -0.42,
  },
  {
    id: 'cs-003',
    timestamp: '2025-04-14 12:15:08',
    description: 'News event with price gap',
    marketRegime: 'Volatile',
    actionTaken: 'Close Position',
    outcome: 'Positive',
    rewardValue: 0.67,
  },
  {
    id: 'cs-004',
    timestamp: '2025-04-14 14:05:31',
    description: 'Trend continuation after pullback',
    marketRegime: 'Trending',
    actionTaken: 'Buy',
    outcome: 'Positive',
    rewardValue: 1.23,
  },
  {
    id: 'cs-005',
    timestamp: '2025-04-14 15:30:45',
    description: 'Range breakdown with increasing volume',
    marketRegime: 'Breakout',
    actionTaken: 'Sell',
    outcome: 'Positive',
    rewardValue: 0.91,
  },
];

// Mock data for current state analysis
const mockStateActions: StateAction[] = [
  {
    state: 'Buy',
    action: 'Open long position',
    value: 0.72,
    confidence: 0.68,
    explanation: 'Strong uptrend detected with increasing volume and bullish engulfing pattern',
  },
  {
    state: 'Sell',
    action: 'Open short position',
    value: 0.45,
    confidence: 0.42,
    explanation: 'Moderate bearish signal with overbought conditions and resistance rejection',
  },
  {
    state: 'Hold',
    action: 'Maintain current position',
    value: 0.65,
    confidence: 0.61,
    explanation: 'Current position aligned with trend direction, no significant reversal signals',
  },
  {
    state: 'Close',
    action: 'Close all positions',
    value: 0.38,
    confidence: 0.35,
    explanation: 'Increased market uncertainty with mixed signals across timeframes',
  },
];

const ModelExplainabilityVisualization: React.FC = () => {
  const [activeTab, setActiveTab] = useState('1');
  const [modelVersion, setModelVersion] = useState('PPO-v1.2.5');
  const [timeframe, setTimeframe] = useState('1m');
  const [loading, setLoading] = useState(false);
  const [selectedCriticalState, setSelectedCriticalState] = useState<CriticalState | null>(null);

  // Columns for feature importance table
  const featureColumns: ColumnsType<FeatureImportance> = [
    {
      title: 'Feature',
      dataIndex: 'feature',
      key: 'feature',
      sorter: (a, b) => a.feature.localeCompare(b.feature),
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => {
        let color = 'blue';
        switch(category) {
          case 'price': color = 'blue'; break;
          case 'volume': color = 'purple'; break;
          case 'indicator': color = 'green'; break;
          case 'market': color = 'orange'; break;
          case 'orderbook': color = 'cyan'; break;
          case 'position': color = 'red'; break;
          case 'time': color = 'magenta'; break;
          default: color = 'default';
        }
        return <Tag color={color}>{category}</Tag>;
      },
      filters: [
        { text: 'Price', value: 'price' },
        { text: 'Volume', value: 'volume' },
        { text: 'Indicator', value: 'indicator' },
        { text: 'Market', value: 'market' },
        { text: 'Orderbook', value: 'orderbook' },
        { text: 'Position', value: 'position' },
        { text: 'Time', value: 'time' },
      ],
      onFilter: (value, record) => record.category === value,
    },
    {
      title: 'Importance',
      dataIndex: 'importance',
      key: 'importance',
      defaultSortOrder: 'descend',
      sorter: (a, b) => a.importance - b.importance,
      render: (importance: number) => {
        // Create a visual bar to represent importance
        const width = `${importance * 100}%`;
        const color = importance > 0.7 ? '#f50' : importance > 0.4 ? '#1890ff' : '#52c41a';
        return (
          <div className="flex items-center">
            <div className="flex-1 mr-2">
              <div style={{ background: '#f0f0f0', height: '14px', borderRadius: '7px', overflow: 'hidden' }}>
                <div style={{ width, background: color, height: '100%' }}></div>
              </div>
            </div>
            <div style={{ width: '36px', textAlign: 'right' }}>{(importance * 100).toFixed(0)}%</div>
          </div>
        );
      },
    },
  ];

  // Columns for critical states table
  const criticalStatesColumns: ColumnsType<CriticalState> = [
    {
      title: 'Timestamp',
      dataIndex: 'timestamp',
      key: 'timestamp',
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: 'Market Regime',
      dataIndex: 'marketRegime',
      key: 'marketRegime',
      render: (regime: string) => {
        let color = 'blue';
        switch(regime) {
          case 'Trending': color = 'green'; break;
          case 'Ranging': color = 'blue'; break;
          case 'Volatile': color = 'volcano'; break;
          case 'Breakout': color = 'purple'; break;
          default: color = 'default';
        }
        return <Tag color={color}>{regime}</Tag>;
      },
    },
    {
      title: 'Action',
      dataIndex: 'actionTaken',
      key: 'actionTaken',
    },
    {
      title: 'Outcome',
      dataIndex: 'outcome',
      key: 'outcome',
      render: (outcome: string) => {
        const color = outcome === 'Positive' ? 'green' : 'red';
        return <Tag color={color}>{outcome}</Tag>;
      },
    },
    {
      title: 'Reward',
      dataIndex: 'rewardValue',
      key: 'rewardValue',
      render: (value: number) => {
        const color = value >= 0 ? 'green' : 'red';
        return <span style={{ color }}>{value.toFixed(2)}</span>;
      },
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button 
          type="link" 
          size="small" 
          icon={<FileSearchOutlined />}
          onClick={() => setSelectedCriticalState(record)}
        >
          Analyze
        </Button>
      ),
    },
  ];

  // Simulate loading data when changing model or timeframe
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 800));
      setLoading(false);
    };
    
    loadData();
  }, [modelVersion, timeframe]);

  return (
    <div className="p-4 bg-gray-50 rounded-lg shadow">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">Model Explainability</h2>

      <div className="bg-white p-4 rounded shadow mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <span className="mr-2">Model Version:</span>
            <Select 
              value={modelVersion} 
              onChange={setModelVersion} 
              style={{ width: 180 }}
            >
              <Option value="PPO-v1.2.5">PPO-v1.2.5</Option>
              <Option value="SAC-v1.1.0">SAC-v1.1.0</Option>
              <Option value="A2C-v0.9.8">A2C-v0.9.8</Option>
              <Option value="DQN-v1.0.3">DQN-v1.0.3</Option>
            </Select>
          </div>
          
          <div className="flex items-center">
            <span className="mr-2">Timeframe:</span>
            <Select 
              value={timeframe} 
              onChange={setTimeframe}
              style={{ width: 120 }}
            >
              <Option value="1m">1 Minute</Option>
              <Option value="5m">5 Minutes</Option>
              <Option value="15m">15 Minutes</Option>
              <Option value="1h">1 Hour</Option>
              <Option value="4h">4 Hours</Option>
            </Select>
          </div>
        </div>
      </div>

      <Spin spinning={loading}>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="Feature Importance" key="1">
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-lg font-medium mb-4">Feature Importance Analysis</h3>
              <p className="text-sm text-gray-600 mb-4">
                This visualization shows which features have the most influence on the model's decisions. 
                Features are ranked by their relative importance score.
              </p>
              
              <Table 
                columns={featureColumns} 
                dataSource={mockFeatureImportance.map((item, index) => ({...item, key: index}))} 
                pagination={false} 
              />
              
              <div className="mt-6">
                <h4 className="font-medium mb-2">Understanding Feature Importance</h4>
                <p className="text-sm text-gray-600">
                  Feature importance is calculated using integrated gradients, which measures the contribution 
                  of each input feature to the model's output. Higher importance means the feature has a 
                  stronger influence on the model's decision-making process.
                </p>
              </div>
            </div>
          </TabPane>
          
          <TabPane tab="State-Action Analysis" key="2">
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-lg font-medium mb-4">Current State-Action Analysis</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {mockStateActions.map((item, index) => (
                  <Card key={index} title={item.state} size="small">
                    <p><strong>Action:</strong> {item.action}</p>
                    <p><strong>Value:</strong> {item.value.toFixed(2)}</p>
                    <p><strong>Confidence:</strong> {(item.confidence * 100).toFixed(0)}%</p>
                    <p><strong>Explanation:</strong> {item.explanation}</p>
                    <div className="mt-2">
                      <div className="bg-gray-100 h-2 w-full rounded-full overflow-hidden">
                        <div 
                          className="bg-blue-500 h-full" 
                          style={{ width: `${item.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Heatmap data={{}} />
                <AttentionVisualization data={{}} />
              </div>
            </div>
          </TabPane>
          
          <TabPane tab="Critical States" key="3">
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-lg font-medium mb-4 flex items-center">
                Critical States Analysis
                <Tooltip title="States where the model made significant decisions or experienced high uncertainty">
                  <QuestionCircleOutlined className="ml-2" />
                </Tooltip>
              </h3>
              
              <Table 
                columns={criticalStatesColumns} 
                dataSource={mockCriticalStates.map(item => ({...item, key: item.id}))} 
                pagination={{ pageSize: 5 }}
              />
              
              {selectedCriticalState && (
                <div className="mt-6 border-t pt-4">
                  <h4 className="font-medium mb-2">Detailed Analysis of State {selectedCriticalState.id}</h4>
                  <div className="bg-gray-50 p-4 rounded">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p><strong>Timestamp:</strong> {selectedCriticalState.timestamp}</p>
                        <p><strong>Description:</strong> {selectedCriticalState.description}</p>
                        <p><strong>Market Regime:</strong> {selectedCriticalState.marketRegime}</p>
                        <p><strong>Action Taken:</strong> {selectedCriticalState.actionTaken}</p>
                      </div>
                      <div>
                        <p><strong>Outcome:</strong> {selectedCriticalState.outcome}</p>
                        <p><strong>Reward Value:</strong> {selectedCriticalState.rewardValue.toFixed(2)}</p>
                        <p><strong>Confidence Level:</strong> {(Math.random() * 30 + 70).toFixed(1)}%</p>
                        <p><strong>Alternative Action:</strong> Hold (Q-value: {(Math.random() * 0.5 + 0.3).toFixed(2)})</p>
                      </div>
                    </div>
                    
                    <Divider />
                    
                    <div className="text-sm">
                      <p><strong>Key Features at Decision Point:</strong></p>
                      <ul className="list-disc pl-5">
                        <li>Price was near a major support level (1.0823)</li>
                        <li>RSI showed oversold conditions (28.5)</li>
                        <li>Volume was 2.3x average</li>
                        <li>Volatility had increased by 45% in the previous 5 minutes</li>
                        <li>Market regime was detected as transitioning from Ranging to Volatile</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </TabPane>
          
          <TabPane tab="Counterfactual Analysis" key="4">
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-lg font-medium mb-4">Counterfactual Exploration</h3>
              <p className="text-sm text-gray-600 mb-4">
                This tool allows you to explore "what if" scenarios by modifying features and seeing how the model's 
                decision would change. Understand how different market conditions would affect the model's behavior.
              </p>
              
              <div className="text-center py-12">
                <p className="text-gray-500">
                  Interactive counterfactual analysis tool would be implemented here, allowing users to adjust 
                  feature values and see how model decisions would change.
                </p>
              </div>
            </div>
          </TabPane>
        </Tabs>
      </Spin>
    </div>
  );
};

export default ModelExplainabilityVisualization;
