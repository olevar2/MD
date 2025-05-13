"""
Advanced Backtesting Reporting Module

This module provides comprehensive reporting capabilities for the backtesting engine,
including performance attribution, visualization components, and exportable formats.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pdfkit
import xlsxwriter
from jinja2 import Environment, FileSystemLoader

class BacktestReport:
    """
    Comprehensive backtesting report generator with performance attribution
    
    This class generates detailed reports for backtest results, including performance
    metrics, trade analysis, market regime analysis, and exportable formats.
    """
    
    def __init__(self, backtest_engine, output_dir: Optional[str] = None):
        """
        Initialize the report generator
        
        Args:
            backtest_engine: The backtesting engine instance with results
            output_dir: Directory to save reports (uses backtest engine's if None)
        """
        self.engine = backtest_engine
        self.logger = logging.getLogger(f"backtest_report.{backtest_engine.backtest_id}")
        
        # Use backtest engine's output directory if none provided
        self.output_dir = output_dir or backtest_engine.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Templates directory
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Initialize Jinja2 environment for report templates
        self.jinja_env = Environment(loader=FileSystemLoader(self.templates_dir))
    
    def generate_performance_report(self, include_trades: bool = True, 
                                    include_drawdowns: bool = True,
                                    include_metrics: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report
        
        Args:
            include_trades: Include detailed trade analysis
            include_drawdowns: Include drawdown analysis
            include_metrics: Include detailed performance metrics
            
        Returns:
            Dictionary containing report data
        """
        self.logger.info("Generating comprehensive performance report")
        
        report_data = {
            "backtest_id": self.engine.backtest_id,
            "timestamp": datetime.now().isoformat(),
            "summary_metrics": self._calculate_summary_metrics(),
            "equity_curve": self._prepare_equity_curve_data(),
        }
        
        # Add detailed trade analysis
        if include_trades:
            report_data["trade_analysis"] = self._analyze_trades()
        
        # Add drawdown analysis
        if include_drawdowns:
            report_data["drawdown_analysis"] = self._analyze_drawdowns()
        
        # Add detailed metrics
        if include_metrics:
            report_data["detailed_metrics"] = self._calculate_detailed_metrics()
        
        # Add performance attribution
        report_data["performance_attribution"] = self._calculate_performance_attribution()
        
        # Add market regime analysis
        report_data["market_regime_analysis"] = self._analyze_market_regimes()
        
        return report_data
    
    def create_interactive_dashboard(self, report_data: Dict[str, Any] = None) -> str:
        """
        Create an interactive HTML dashboard for backtest results
        
        Args:
            report_data: Report data (generated if not provided)
            
        Returns:
            Path to the generated HTML dashboard
        """
        self.logger.info("Creating interactive dashboard")
        
        # Generate report data if not provided
        if report_data is None:
            report_data = self.generate_performance_report()
        
        # Create dashboard with plotly
        dashboard = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Equity Curve", "Drawdown Analysis", 
                "Monthly Returns", "Trade Distribution",
                "Performance Attribution", "Market Regime Performance"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "bar"}]
            ],
            vertical_spacing=0.1
        )
        
        # Add equity curve
        equity_data = report_data["equity_curve"]
        dashboard.add_trace(
            go.Scatter(x=equity_data["dates"], y=equity_data["equity"], 
                      name="Equity", line=dict(color="blue")),
            row=1, col=1
        )
        
        # Add drawdown analysis
        drawdown_data = report_data["drawdown_analysis"]
        dashboard.add_trace(
            go.Scatter(x=drawdown_data["dates"], y=drawdown_data["drawdown_percentage"], 
                      name="Drawdown %", line=dict(color="red")),
            row=1, col=2
        )
        
        # Add monthly returns
        monthly_returns = report_data["detailed_metrics"]["monthly_returns"]
        dashboard.add_trace(
            go.Bar(x=list(monthly_returns.keys()), y=list(monthly_returns.values()),
                  name="Monthly Returns"),
            row=2, col=1
        )
        
        # Add trade distribution
        trade_data = report_data["trade_analysis"]
        dashboard.add_trace(
            go.Histogram(x=trade_data["trade_returns"], name="Trade Returns"),
            row=2, col=2
        )
        
        # Add performance attribution
        attribution = report_data["performance_attribution"]
        dashboard.add_trace(
            go.Pie(labels=list(attribution.keys()), values=list(attribution.values()),
                  name="Attribution"),
            row=3, col=1
        )
        
        # Add market regime performance
        regime_data = report_data["market_regime_analysis"]
        dashboard.add_trace(
            go.Bar(x=list(regime_data.keys()), y=list(regime_data.values()),
                  name="Regime Performance"),
            row=3, col=2
        )
        
        # Update layout
        dashboard.update_layout(
            title_text=f"Backtest Results: {self.engine.backtest_id}",
            height=900,
            width=1200,
            showlegend=True
        )
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, f"{self.engine.backtest_id}_dashboard.html")
        dashboard.write_html(dashboard_path)
        self.logger.info(f"Interactive dashboard saved to {dashboard_path}")
        
        return dashboard_path
    
    def export_pdf_report(self, report_data: Dict[str, Any] = None) -> str:
        """
        Export backtest results to a PDF report
        
        Args:
            report_data: Report data (generated if not provided)
            
        Returns:
            Path to the exported PDF report
        """
        self.logger.info("Exporting PDF report")
        
        # Generate report data if not provided
        if report_data is None:
            report_data = self.generate_performance_report()
        
        # Load template
        template = self.jinja_env.get_template("pdf_report_template.html")
        
        # Render template with report data
        html_content = template.render(
            backtest_id=self.engine.backtest_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_metrics=report_data["summary_metrics"],
            trade_analysis=report_data["trade_analysis"],
            drawdown_analysis=report_data["drawdown_analysis"],
            performance_attribution=report_data["performance_attribution"],
            market_regime_analysis=report_data["market_regime_analysis"]
        )
        
        # Save HTML version
        html_path = os.path.join(self.output_dir, f"{self.engine.backtest_id}_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Convert to PDF
        pdf_path = os.path.join(self.output_dir, f"{self.engine.backtest_id}_report.pdf")
        pdfkit.from_file(html_path, pdf_path)
        self.logger.info(f"PDF report exported to {pdf_path}")
        
        return pdf_path
    
    def export_excel_report(self, report_data: Dict[str, Any] = None) -> str:
        """
        Export backtest results to an Excel report
        
        Args:
            report_data: Report data (generated if not provided)
            
        Returns:
            Path to the exported Excel report
        """
        self.logger.info("Exporting Excel report")
        
        # Generate report data if not provided
        if report_data is None:
            report_data = self.generate_performance_report()
        
        # Create Excel writer
        excel_path = os.path.join(self.output_dir, f"{self.engine.backtest_id}_report.xlsx")
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        
        # Create summary sheet
        pd.DataFrame([report_data["summary_metrics"]]).to_excel(writer, sheet_name="Summary", index=False)
        
        # Create trade analysis sheet
        pd.DataFrame(report_data["trade_analysis"]["trades"]).to_excel(writer, sheet_name="Trades", index=False)
        
        # Create monthly returns sheet
        monthly_returns = pd.DataFrame({
            "Month": list(report_data["detailed_metrics"]["monthly_returns"].keys()),
            "Return": list(report_data["detailed_metrics"]["monthly_returns"].values())
        })
        monthly_returns.to_excel(writer, sheet_name="Monthly Returns", index=False)
        
        # Create performance attribution sheet
        attribution = pd.DataFrame({
            "Factor": list(report_data["performance_attribution"].keys()),
            "Contribution": list(report_data["performance_attribution"].values())
        })
        attribution.to_excel(writer, sheet_name="Performance Attribution", index=False)
        
        # Create market regime sheet
        regimes = pd.DataFrame({
            "Regime": list(report_data["market_regime_analysis"].keys()),
            "Performance": list(report_data["market_regime_analysis"].values())
        })
        regimes.to_excel(writer, sheet_name="Market Regimes", index=False)
        
        # Save Excel file
        writer.save()
        self.logger.info(f"Excel report exported to {excel_path}")
        
        return excel_path
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary performance metrics"""
        metrics = {}
        
        # Use metrics from backtest engine if available
        if hasattr(self.engine, "metrics") and self.engine.metrics:
            metrics.update(self.engine.metrics)
        
        # Calculate additional metrics if needed
        if "total_return_pct" not in metrics:
            final_equity = self.engine.balance_history[-1] if self.engine.balance_history else self.engine.balance
            metrics["total_return_pct"] = ((final_equity / self.engine.initial_balance) - 1) * 100
        
        if "win_rate" not in metrics:
            profitable_trades = sum(1 for p in self.engine.closed_positions if p["pnl"] > 0)
            total_trades = len(self.engine.closed_positions)
            metrics["win_rate"] = profitable_trades / total_trades if total_trades > 0 else 0
            
        if "sharpe_ratio" not in metrics:
            metrics["sharpe_ratio"] = self._calculate_sharpe_ratio()
            
        if "max_drawdown" not in metrics:
            metrics["max_drawdown"] = self._calculate_max_drawdown()
            
        return metrics
    
    def _prepare_equity_curve_data(self) -> Dict[str, List]:
        """Prepare equity curve data for plotting"""
        equity_data = {
            "dates": [],
            "equity": [],
            "balance": []
        }
        
        # Process equity history
        for entry in self.engine.equity_history:
            equity_data["dates"].append(entry["timestamp"])
            equity_data["equity"].append(entry["equity"])
            equity_data["balance"].append(entry["balance"])
        
        return equity_data
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade performance"""
        trades = []
        returns = []
        durations = []
        
        for position in self.engine.closed_positions:
            trade_return = position["pnl"] / position["size"] if position["size"] > 0 else 0
            returns.append(trade_return)
            
            # Calculate trade duration
            duration = (position["close_time"] - position["open_time"]).total_seconds() / 3600  # hours
            durations.append(duration)
            
            trades.append({
                "id": position["id"],
                "direction": position["direction"],
                "entry_price": position["entry_price"],
                "exit_price": position["exit_price"],
                "size": position["size"],
                "pnl": position["pnl"],
                "return": trade_return,
                "open_time": position["open_time"],
                "close_time": position["close_time"],
                "duration_hours": duration
            })
        
        return {
            "trades": trades,
            "trade_returns": returns,
            "trade_durations": durations,
            "avg_return": np.mean(returns) if returns else 0,
            "avg_duration": np.mean(durations) if durations else 0,
            "best_trade": max(returns) if returns else 0,
            "worst_trade": min(returns) if returns else 0
        }
    
    def _analyze_drawdowns(self) -> Dict[str, Any]:
        """Analyze drawdowns in equity curve"""
        equity = [entry["equity"] for entry in self.engine.equity_history]
        dates = [entry["timestamp"] for entry in self.engine.equity_history]
        
        if not equity:
            return {
                "max_drawdown": 0,
                "max_drawdown_duration": 0,
                "drawdown_periods": [],
                "dates": [],
                "drawdown_percentage": []
            }
            
        # Calculate drawdown series
        peak = equity[0]
        drawdown_pct = [0]
        drawdown_abs = [0]
        
        for i in range(1, len(equity)):
            peak = max(peak, equity[i])
            dd_abs = equity[i] - peak
            dd_pct = (equity[i] / peak - 1) * 100 if peak > 0 else 0
            drawdown_abs.append(dd_abs)
            drawdown_pct.append(dd_pct)
        
        # Find drawdown periods
        in_drawdown = False
        drawdown_start = 0
        drawdown_periods = []
        
        for i in range(len(drawdown_pct)):
            if not in_drawdown and drawdown_pct[i] < 0:
                in_drawdown = True
                drawdown_start = i
            elif in_drawdown and drawdown_pct[i] >= 0:
                in_drawdown = False
                drawdown_periods.append({
                    "start_date": dates[drawdown_start],
                    "end_date": dates[i],
                    "duration": i - drawdown_start,
                    "max_drawdown": min(drawdown_pct[drawdown_start:i+1])
                })
        
        # If still in drawdown at the end
        if in_drawdown:
            drawdown_periods.append({
                "start_date": dates[drawdown_start],
                "end_date": dates[-1],
                "duration": len(dates) - drawdown_start,
                "max_drawdown": min(drawdown_pct[drawdown_start:])
            })
        
        return {
            "max_drawdown": min(drawdown_pct) if drawdown_pct else 0,
            "max_drawdown_duration": max([d["duration"] for d in drawdown_periods]) if drawdown_periods else 0,
            "drawdown_periods": drawdown_periods,
            "dates": dates,
            "drawdown_percentage": drawdown_pct
        }
    
    def _calculate_detailed_metrics(self) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        # Monthly returns
        monthly_returns = {}
        for position in self.engine.closed_positions:
            month_key = position["close_time"].strftime("%Y-%m")
            if month_key not in monthly_returns:
                monthly_returns[month_key] = 0
            monthly_returns[month_key] += position["pnl"]
            
        # Profit factor
        gross_profit = sum(p["pnl"] for p in self.engine.closed_positions if p["pnl"] > 0)
        gross_loss = sum(p["pnl"] for p in self.engine.closed_positions if p["pnl"] < 0)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else float('inf')
        
        # Return metrics
        if self.engine.equity_history:
            equity = [entry["equity"] for entry in self.engine.equity_history]
            returns = [(equity[i] / equity[i-1]) - 1 for i in range(1, len(equity))]
            avg_return = np.mean(returns) if returns else 0
            volatility = np.std(returns) if returns else 0
            
            # Calculate Sharpe/Sortino ratios
            sharpe = (avg_return / volatility) * np.sqrt(252) if volatility > 0 else 0
            
            # Calculate downside deviation for Sortino
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            sortino = (avg_return / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        else:
            avg_return = 0
            volatility = 0
            sharpe = 0
            sortino = 0
        
        # Recovery factor
        max_dd = self._calculate_max_drawdown()
        total_return = ((self.engine.balance / self.engine.initial_balance) - 1)
        recovery_factor = abs(total_return / max_dd) if max_dd < 0 else float('inf')
        
        return {
            "monthly_returns": monthly_returns,
            "profit_factor": profit_factor,
            "recovery_factor": recovery_factor,
            "average_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": abs(total_return / max_dd) * 252 if max_dd < 0 else float('inf')
        }
    
    def _calculate_performance_attribution(self) -> Dict[str, float]:
        """
        Calculate performance attribution across different factors
        
        This breaks down the total return into components attributed to:
        - Market regime (trend, range, volatility)
        - Trade direction (long vs short)
        - Trade duration (short-term vs long-term)
        - Technical tool signals
        """
        total_pnl = sum(p["pnl"] for p in self.engine.closed_positions)
        if total_pnl == 0:
            return {"No trades": 100}
        
        attribution = {}
        
        # Attribution by trade direction
        long_pnl = sum(p["pnl"] for p in self.engine.closed_positions if p["direction"] == "long")
        short_pnl = sum(p["pnl"] for p in self.engine.closed_positions if p["direction"] == "short")
        
        attribution["Long trades"] = (long_pnl / total_pnl) * 100 if total_pnl != 0 else 0
        attribution["Short trades"] = (short_pnl / total_pnl) * 100 if total_pnl != 0 else 0
        
        # Attribution by trade duration
        short_term_pnl = sum(p["pnl"] for p in self.engine.closed_positions 
                            if (p["close_time"] - p["open_time"]).total_seconds() < 24*3600)
        long_term_pnl = sum(p["pnl"] for p in self.engine.closed_positions 
                           if (p["close_time"] - p["open_time"]).total_seconds() >= 24*3600)
        
        attribution["Short-term trades"] = (short_term_pnl / total_pnl) * 100 if total_pnl != 0 else 0
        attribution["Long-term trades"] = (long_term_pnl / total_pnl) * 100 if total_pnl != 0 else 0
        
        # Attribution by technical tool if available
        if hasattr(self.engine, "tool_evaluator") and self.engine.tool_evaluator:
            tool_pnls = {}
            for position in self.engine.closed_positions:
                if "tool" in position:
                    tool = position["tool"]
                    if tool not in tool_pnls:
                        tool_pnls[tool] = 0
                    tool_pnls[tool] += position["pnl"]
            
            for tool, pnl in tool_pnls.items():
                attribution[f"Tool: {tool}"] = (pnl / total_pnl) * 100 if total_pnl != 0 else 0
        
        return attribution
    
    def _analyze_market_regimes(self) -> Dict[str, float]:
        """Analyze performance across different market regimes"""
        if not hasattr(self.engine, "market_regimes") or not self.engine.market_regimes:
            return {"Unknown": 100}
        
        regime_pnls = {}
        for position in self.engine.closed_positions:
            # Find the regime at position open time
            regime = "Unknown"
            for r in self.engine.market_regimes:
                if r["start_time"] <= position["open_time"] <= r["end_time"]:
                    regime = r["regime"]
                    break
                    
            if regime not in regime_pnls:
                regime_pnls[regime] = 0
            regime_pnls[regime] += position["pnl"]
        
        total_pnl = sum(position["pnl"] for position in self.engine.closed_positions)
        
        # Calculate percentage contribution
        regime_performance = {}
        for regime, pnl in regime_pnls.items():
            regime_performance[regime] = (pnl / total_pnl) * 100 if total_pnl != 0 else 0
            
        return regime_performance
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if not self.engine.equity_history:
            return 0
            
        equity = [entry["equity"] for entry in self.engine.equity_history]
        returns = [(equity[i] / equity[i-1]) - 1 for i in range(1, len(equity))]
        
        if not returns:
            return 0
            
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualized Sharpe ratio (assuming daily equity points)
        risk_free = 0.01 / 252  # 1% annual risk-free rate
        sharpe = (avg_return - risk_free) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        return sharpe
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.engine.equity_history:
            return 0
            
        equity = [entry["equity"] for entry in self.engine.equity_history]
        peak = equity[0]
        max_dd = 0
        
        for e in equity[1:]:
            if e > peak:
                peak = e
            dd = (e / peak - 1)
            if dd < max_dd:
                max_dd = dd
                
        return max_dd * 100  # convert to percentage
