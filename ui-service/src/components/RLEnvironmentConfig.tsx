import React, { useState, useEffect } from 'react';
import { Slider, Select, Switch, Input, Button, Tooltip, Collapse, Divider, InputNumber, Tag, Space, Card, Tabs } from 'antd';
import { QuestionCircleOutlined, PlusOutlined, SaveOutlined, LoadOutlined, SettingOutlined } from '@ant-design/icons';
import type { SelectProps } from 'antd';

const { Panel } = Collapse;
const { TabPane } = Tabs;
const { Option } = Select;

interface RewardComponentConfig {
  name: string;
  enabled: boolean;
  weight: number;
  description: string;
}

interface EnvironmentConfig {
  id?: string;
  name: string;
  symbol: string;
  timeframes: string[];
  lookback_periods: number;
  features: string[];
  position_sizing_type: string;
  max_position_size: number;
  trading_fee_percent: number;
  reward_mode: string;
  risk_free_rate: number;
  episode_timesteps: number;
  time_step_seconds: number;
  random_episode_start: boolean;
  curriculum_level: number;
  include_broker_state: boolean;
  include_order_book: boolean;
  include_technical_indicators: boolean;
  include_news_sentiment: boolean;
  observation_normalization: boolean;
  reward_components: RewardComponentConfig[];
  tags: string[];
}

const defaultConfig: EnvironmentConfig = {
  name: 'Default Environment',
  symbol: 'EUR/USD',
  timeframes: ['1m', '5m', '15m'],
  lookback_periods: 50,
  features: ['open', 'high', 'low', 'close', 'volume', 'spread'],
  position_sizing_type: 'fixed', // "fixed", "dynamic", "risk_based"
  max_position_size: 1.0,
  trading_fee_percent: 0.002,
  reward_mode: 'risk_adjusted', // "pnl", "risk_adjusted", "custom"
  risk_free_rate: 0.02,
  episode_timesteps: 1000,
  time_step_seconds: 60,
  random_episode_start: true,
  curriculum_level: 0,
  include_broker_state: true,
  include_order_book: true,
  include_technical_indicators: true,
  include_news_sentiment: true,
  observation_normalization: true,
  reward_components: [
    { name: 'pnl', enabled: true, weight: 1.0, description: 'Reward from realized and unrealized PnL' },
    { name: 'volatility_penalty', enabled: true, weight: -0.2, description: 'Penalty for excessive return volatility' },
    { name: 'drawdown_penalty', enabled: true, weight: -0.3, description: 'Penalty for significant drawdowns' },
    { name: 'trade_frequency_penalty', enabled: true, weight: -0.1, description: 'Penalty for excessive trading' },
    { name: 'news_adaptation_bonus', enabled: true, weight: 0.3, description: 'Bonus for appropriate adaptation to news events' },
    { name: 'risk_reward_bonus', enabled: false, weight: 0.2, description: 'Bonus for good risk-reward ratio trades' },
    { name: 'regime_adaptation_bonus', enabled: false, weight: 0.3, description: 'Bonus for adapting to market regime changes' }
  ],
  tags: ['default', 'basic', 'news-aware']
};

// Available options for selections
const symbolOptions = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CAD', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY'];
const timeframeOptions = ['1s', '5s', '15s', '30s', '1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M'];
const featureOptions = ['open', 'high', 'low', 'close', 'volume', 'spread', 'ask', 'bid', 'mid', 'vwap', 'atr', 'volatility'];
const positionSizingOptions = ['fixed', 'dynamic', 'risk_based', 'position_sizer', 'adaptive'];
const rewardModeOptions = ['pnl', 'risk_adjusted', 'sharpe', 'sortino', 'omega', 'custom'];

const RLEnvironmentConfig: React.FC = () => {
  const [config, setConfig] = useState<EnvironmentConfig>(defaultConfig);
  const [savedConfigs, setSavedConfigs] = useState<{id: string, name: string}[]>([]);
  const [configName, setConfigName] = useState('');
  const [newTag, setNewTag] = useState('');
  const [activeTab, setActiveTab] = useState('1');

  // Simulate fetching saved configurations from API
  useEffect(() => {
    // Simulated API call - replace with actual API call
    const fetchConfigs = async () => {
      try {
        // Mocked response
        const mockConfigs = [
          { id: '1', name: 'Trending Market Config' },
          { id: '2', name: 'Volatile Market Config' },
          { id: '3', name: 'Range-bound Config' }
        ];
        setSavedConfigs(mockConfigs);
      } catch (error) {
        console.error('Failed to fetch saved configurations:', error);
      }
    };
    
    fetchConfigs();
  }, []);

  const handleSaveConfig = async () => {
    try {
      // Simulated API call - replace with actual API call
      console.log('Saving configuration:', config);
      // Add to saved configs with a fake ID
      const newId = String(savedConfigs.length + 1);
      setSavedConfigs([...savedConfigs, { id: newId, name: config.name }]);
      // Show success message
      alert('Configuration saved successfully!');
    } catch (error) {
      console.error('Failed to save configuration:', error);
      alert('Failed to save configuration. Please try again.');
    }
  };

  const handleLoadConfig = async (id: string) => {
    try {
      // Simulated API call - replace with actual API call
      console.log('Loading configuration with ID:', id);
      
      // Mock different configurations
      let loadedConfig;
      
      if (id === '1') {  // Trending Market
        loadedConfig = {
          ...defaultConfig,
          name: 'Trending Market Config',
          reward_components: [
            ...defaultConfig.reward_components,
            { name: 'trend_following_bonus', enabled: true, weight: 0.4, description: 'Bonus for following the trend' }
          ],
          tags: ['trending', 'momentum']
        };
      } else if (id === '2') {  // Volatile Market
        loadedConfig = {
          ...defaultConfig,
          name: 'Volatile Market Config',
          lookback_periods: 100,
          trading_fee_percent: 0.003,
          reward_components: defaultConfig.reward_components.map(comp => 
            comp.name === 'volatility_penalty' 
              ? { ...comp, weight: -0.4 } 
              : comp
          ),
          tags: ['volatile', 'high-risk']
        };
      } else {  // Range-bound
        loadedConfig = {
          ...defaultConfig,
          name: 'Range-bound Config',
          reward_components: [
            ...defaultConfig.reward_components,
            { name: 'mean_reversion_bonus', enabled: true, weight: 0.35, description: 'Bonus for mean reversion trades' }
          ],
          tags: ['range-bound', 'mean-reversion']
        };
      }
      
      setConfig(loadedConfig);
      setConfigName(loadedConfig.name);
    } catch (error) {
      console.error('Failed to load configuration:', error);
      alert('Failed to load configuration. Please try again.');
    }
  };

  const handleRewardComponentChange = (index: number, field: keyof RewardComponentConfig, value: any) => {
    const newComponents = [...config.reward_components];
    newComponents[index] = { ...newComponents[index], [field]: value };
    setConfig({ ...config, reward_components: newComponents });
  };

  const handleAddRewardComponent = () => {
    const newComponent: RewardComponentConfig = {
      name: `custom_reward_${config.reward_components.length + 1}`,
      enabled: true,
      weight: 0.1,
      description: 'Custom reward component'
    };
    setConfig({
      ...config,
      reward_components: [...config.reward_components, newComponent]
    });
  };

  const handleRemoveRewardComponent = (index: number) => {
    const newComponents = [...config.reward_components];
    newComponents.splice(index, 1);
    setConfig({ ...config, reward_components: newComponents });
  };

  const handleAddTag = () => {
    if (newTag && !config.tags.includes(newTag)) {
      setConfig({ ...config, tags: [...config.tags, newTag] });
      setNewTag('');
    }
  };

  const handleRemoveTag = (tag: string) => {
    setConfig({ ...config, tags: config.tags.filter(t => t !== tag) });
  };

  // Calculate observation space dimensionality (simplified estimation)
  const calculateObservationSpace = () => {
    let dimensions = 0;
    
    // Market data features across timeframes
    if (config.features.length > 0) {
      dimensions += config.features.length * config.timeframes.length * config.lookback_periods;
    }
    
    // Technical indicators (rough estimation)
    if (config.include_technical_indicators) {
      dimensions += 10 * config.timeframes.length; // Assuming ~10 indicators per timeframe
    }
    
    // Order book data
    if (config.include_order_book) {
      dimensions += 20; // Assuming 5 levels with bid/ask price and volume
    }
    
    // Broker state
    if (config.include_broker_state) {
      dimensions += 10; // Various broker state metrics
    }

    // News and Sentiment data
    if (config.include_news_sentiment) {
      dimensions += 15; // 5 impact + 6 event type + 4 impact level features
    }
    
    // Position and account state
    dimensions += 5; // Basic position information
    
    return dimensions;
  };

  return (
    <div className="p-4 bg-gray-50 rounded-lg shadow">
      <h2 className="text-2xl font-semibold mb-4 text-gray-800">RL Environment Configuration</h2>

      <Tabs activeKey={activeTab} onChange={(key) => setActiveTab(key)}>
        <TabPane tab="Environment Settings" key="1">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Column: Basic Settings */}
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-xl font-medium mb-4">Basic Settings</h3>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Environment Name</label>
                <Input 
                  value={config.name} 
                  onChange={(e) => setConfig({ ...config, name: e.target.value })}
                  placeholder="Enter a descriptive name" 
                />
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
                <Select
                  style={{ width: '100%' }}
                  value={config.symbol}
                  onChange={(value) => setConfig({ ...config, symbol: value })}
                >
                  {symbolOptions.map(symbol => (
                    <Option key={symbol} value={symbol}>{symbol}</Option>
                  ))}
                </Select>
              </div>
              
              <div className="mb-4">
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  Timeframes
                  <Tooltip title="Multiple timeframes for multi-timeframe analysis">
                    <QuestionCircleOutlined className="ml-1" />
                  </Tooltip>
                </label>
                <Select
                  mode="multiple"
                  allowClear
                  style={{ width: '100%' }}
                  placeholder="Select timeframes"
                  value={config.timeframes}
                  onChange={(value) => setConfig({ ...config, timeframes: value })}
                  maxTagCount="responsive"
                >
                  {timeframeOptions.map(tf => (
                    <Option key={tf} value={tf}>{tf}</Option>
                  ))}
                </Select>
              </div>
              
              <div className="mb-4">
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  Lookback Periods
                  <Tooltip title="Number of historical periods to include in each observation">
                    <QuestionCircleOutlined className="ml-1" />
                  </Tooltip>
                </label>
                <Slider
                  min={10}
                  max={200}
                  value={config.lookback_periods}
                  onChange={(value) => setConfig({ ...config, lookback_periods: value })}
                />
                <div className="text-right text-xs text-gray-500">{config.lookback_periods} periods</div>
              </div>
              
              <div className="mb-4">
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  Features
                  <Tooltip title="Data features to include in observations">
                    <QuestionCircleOutlined className="ml-1" />
                  </Tooltip>
                </label>
                <Select
                  mode="multiple"
                  allowClear
                  style={{ width: '100%' }}
                  placeholder="Select features"
                  value={config.features}
                  onChange={(value) => setConfig({ ...config, features: value })}
                  maxTagCount="responsive"
                >
                  {featureOptions.map(feature => (
                    <Option key={feature} value={feature}>{feature}</Option>
                  ))}
                </Select>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Episode Length</label>
                <InputNumber
                  style={{ width: '100%' }}
                  min={100}
                  max={10000}
                  step={100}
                  value={config.episode_timesteps}
                  onChange={(value) => setConfig({ ...config, episode_timesteps: value as number })}
                />
                <div className="text-xs text-gray-500 mt-1">Timesteps per episode</div>
              </div>
              
              <div className="mb-4">
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  Random Episode Start
                  <Tooltip title="Randomly select starting points in the data for each episode">
                    <QuestionCircleOutlined className="ml-1" />
                  </Tooltip>
                </label>
                <Switch
                  checked={config.random_episode_start}
                  onChange={(checked) => setConfig({ ...config, random_episode_start: checked })}
                />
              </div>
            </div>

            {/* Right Column: Advanced Settings */}
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-xl font-medium mb-4">Advanced Settings</h3>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Position Sizing Type</label>
                <Select
                  style={{ width: '100%' }}
                  value={config.position_sizing_type}
                  onChange={(value) => setConfig({ ...config, position_sizing_type: value })}
                >
                  {positionSizingOptions.map(option => (
                    <Option key={option} value={option}>{option}</Option>
                  ))}
                </Select>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Max Position Size (Lots)</label>
                <Slider
                  min={0.01}
                  max={10}
                  step={0.01}
                  value={config.max_position_size}
                  onChange={(value) => setConfig({ ...config, max_position_size: value })}
                />
                <div className="text-right text-xs text-gray-500">{config.max_position_size} lots</div>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Trading Fee (%)</label>
                <Slider
                  min={0}
                  max={0.01}
                  step={0.0001}
                  value={config.trading_fee_percent}
                  onChange={(value) => setConfig({ ...config, trading_fee_percent: value })}
                />
                <div className="text-right text-xs text-gray-500">{(config.trading_fee_percent * 100).toFixed(4)}%</div>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Reward Mode</label>
                <Select
                  style={{ width: '100%' }}
                  value={config.reward_mode}
                  onChange={(value) => setConfig({ ...config, reward_mode: value })}
                >
                  {rewardModeOptions.map(option => (
                    <Option key={option} value={option}>{option}</Option>
                  ))}
                </Select>
              </div>
              
              <div className="mb-4">
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  Curriculum Level
                  <Tooltip title="Higher levels introduce more complexity and challenges">
                    <QuestionCircleOutlined className="ml-1" />
                  </Tooltip>
                </label>
                <Slider
                  min={0}
                  max={5}
                  marks={{ 0: 'Easiest', 5: 'Hardest' }}
                  value={config.curriculum_level}
                  onChange={(value) => setConfig({ ...config, curriculum_level: value })}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-2 mt-6">
                <div className="flex items-center">
                  <Switch
                    checked={config.include_broker_state}
                    onChange={(checked) => setConfig({ ...config, include_broker_state: checked })}
                  />
                  <span className="ml-2 text-sm">Include Broker State</span>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={config.include_order_book}
                    onChange={(checked) => setConfig({ ...config, include_order_book: checked })}
                  />
                  <span className="ml-2 text-sm">Include Order Book</span>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={config.include_technical_indicators}
                    onChange={(checked) => setConfig({ ...config, include_technical_indicators: checked })}
                  />
                  <span className="ml-2 text-sm">Include Technical Indicators</span>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={config.include_news_sentiment}
                    onChange={(checked) => setConfig({ ...config, include_news_sentiment: checked })}
                  />
                  <span className="ml-2 text-sm">Include News/Sentiment</span>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={config.observation_normalization}
                    onChange={(checked) => setConfig({ ...config, observation_normalization: checked })}
                  />
                  <span className="ml-2 text-sm">Normalize Observations</span>
                </div>
              </div>
              
              <div className="mt-6">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-700">Tags:</span>
                  <div className="flex items-center">
                    <Input
                      size="small"
                      value={newTag}
                      onChange={(e) => setNewTag(e.target.value)}
                      onPressEnter={handleAddTag}
                      placeholder="Add tag"
                      style={{ width: 120 }}
                    />
                    <Button size="small" icon={<PlusOutlined />} onClick={handleAddTag} className="ml-1" />
                  </div>
                </div>
                <div className="mt-2">
                  {config.tags.map(tag => (
                    <Tag 
                      key={tag} 
                      closable 
                      onClose={() => handleRemoveTag(tag)}
                      className="mb-1 mr-1"
                    >
                      {tag}
                    </Tag>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          {/* Reward Components */}
          <div className="bg-white p-4 rounded shadow mt-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-medium">Reward Components</h3>
              <Button 
                type="primary" 
                icon={<PlusOutlined />} 
                onClick={handleAddRewardComponent}
              >
                Add Component
              </Button>
            </div>
            
            {config.reward_components.map((component, index) => (
              <div key={index} className="border rounded p-4 mb-4">
                <div className="flex justify-between items-start">
                  <div className="flex-1 mr-4">
                    <Input
                      value={component.name}
                      onChange={(e) => handleRewardComponentChange(index, 'name', e.target.value)}
                      placeholder="Component name"
                      className="mb-2"
                    />
                    <Input
                      value={component.description}
                      onChange={(e) => handleRewardComponentChange(index, 'description', e.target.value)}
                      placeholder="Description"
                    />
                  </div>
                  <div className="w-24 mr-2">
                    <label className="block text-xs font-medium text-gray-700">Weight</label>
                    <InputNumber
                      value={component.weight}
                      onChange={(value) => handleRewardComponentChange(index, 'weight', value)}
                      step={0.1}
                      style={{ width: '100%' }}
                    />
                  </div>
                  <div className="flex items-center">
                    <Switch
                      checked={component.enabled}
                      onChange={(checked) => handleRewardComponentChange(index, 'enabled', checked)}
                      disabled={component.name === 'pnl'} // Ensure PnL is always enabled
                    />
                    <span className="ml-1 mr-4 text-sm">Enabled</span>
                    <Button 
                      danger 
                      size="small"
                      onClick={() => handleRemoveRewardComponent(index)}
                      disabled={['pnl', 'volatility_penalty', 'drawdown_penalty', 'trade_frequency_penalty', 'news_adaptation_bonus'].includes(component.name)} // Prevent removing core components
                    >
                      Remove
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* Environment Summary */}
          <div className="bg-white p-4 rounded shadow mt-6">
            <h3 className="text-xl font-medium mb-2">Environment Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p><strong>Observation Space Dimensionality:</strong> ~{calculateObservationSpace()} dimensions</p>
                <p><strong>Action Space:</strong> 4 dimensions (action type, position size, SL, TP)</p>
                <p><strong>Total Training Steps:</strong> {config.episode_timesteps.toLocaleString()} steps per episode</p>
              </div>
              <div>
                <p><strong>Active Reward Components:</strong> {config.reward_components.filter(c => c.enabled).length}</p>
                <p><strong>News/Sentiment Features:</strong> {config.include_news_sentiment ? 'Enabled (15 features)' : 'Disabled'}</p>
                <p><strong>Total Features:</strong> {config.features.length * config.timeframes.length} (across {config.timeframes.length} timeframes)</p>
                <p><strong>Memory Requirements:</strong> {(calculateObservationSpace() * 4 * config.episode_timesteps / (1024 * 1024)).toFixed(2)} MB (estimated)</p>
              </div>
            </div>
          </div>
        </TabPane>
        
        <TabPane tab="Load & Save" key="2">
          <div className="bg-white p-4 rounded shadow">
            <h3 className="text-xl font-medium mb-4">Save Configuration</h3>
            <div className="mb-4 flex">
              <Input
                value={config.name}
                onChange={(e) => setConfig({ ...config, name: e.target.value })}
                placeholder="Configuration name"
                className="flex-1 mr-4"
              />
              <Button 
                type="primary" 
                icon={<SaveOutlined />} 
                onClick={handleSaveConfig}
                disabled={!config.name.trim()}
              >
                Save Configuration
              </Button>
            </div>
            
            <Divider />
            
            <h3 className="text-xl font-medium mb-4">Load Configuration</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {savedConfigs.map(savedConfig => (
                <Card 
                  key={savedConfig.id} 
                  size="small" 
                  title={savedConfig.name}
                  extra={<Button 
                    size="small" 
                    type="primary" 
                    icon={<LoadOutlined />} 
                    onClick={() => handleLoadConfig(savedConfig.id)}
                  />}
                  className="hover:shadow-md transition-shadow"
                >
                  <p className="text-xs text-gray-500">ID: {savedConfig.id}</p>
                  <p className="text-xs text-gray-500">Market Simulation</p>
                </Card>
              ))}
            </div>
          </div>
        </TabPane>
      </Tabs>

      <div className="mt-6 flex justify-end">
        <Button type="default" className="mr-2">Cancel</Button>
        <Button 
          type="primary" 
          icon={<SettingOutlined />}
          onClick={() => alert('This would create the environment with the current configuration!')}
        >
          Create Environment
        </Button>
      </div>
    </div>
  );
};

export default RLEnvironmentConfig;
