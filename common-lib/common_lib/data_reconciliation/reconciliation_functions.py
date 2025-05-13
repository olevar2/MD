#!/usr/bin/env python3
"""
Reconciliation functions for different data types.

This module provides specific reconciliation functions for
different types of data in the Forex Trading Platform.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def reconcile_orders(source_data: List[Dict[str, Any]], target_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconcile orders between services.
    
    Args:
        source_data: Orders from the source service
        target_data: Orders from the target service
        
    Returns:
        Reconciliation results
    """
    source_orders = {order['id']: order for order in source_data}
    target_orders = {order['id']: order for order in target_data}
    
    # Find missing orders
    missing_in_target = [order_id for order_id in source_orders if order_id not in target_orders]
    missing_in_source = [order_id for order_id in target_orders if order_id not in source_orders]
    
    # Find mismatched orders
    mismatched = []
    for order_id in set(source_orders.keys()) & set(target_orders.keys()):
        source_order = source_orders[order_id]
        target_order = target_orders[order_id]
        
        # Compare relevant fields
        if source_order['status'] != target_order['status'] or            source_order['quantity'] != target_order['quantity'] or            source_order['price'] != target_order['price']:
            mismatched.append({
                'order_id': order_id,
                'source': source_order,
                'target': target_order
            })
    
    return {
        'total_source_orders': len(source_data),
        'total_target_orders': len(target_data),
        'missing_in_target': missing_in_target,
        'missing_in_source': missing_in_source,
        'mismatched': mismatched,
        'is_consistent': len(missing_in_target) == 0 and len(missing_in_source) == 0 and len(mismatched) == 0
    }

def reconcile_positions(source_data: List[Dict[str, Any]], target_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconcile positions between services.
    
    Args:
        source_data: Positions from the source service
        target_data: Positions from the target service
        
    Returns:
        Reconciliation results
    """
    source_positions = {position['id']: position for position in source_data}
    target_positions = {position['id']: position for position in target_data}
    
    # Find missing positions
    missing_in_target = [position_id for position_id in source_positions if position_id not in target_positions]
    missing_in_source = [position_id for position_id in target_positions if position_id not in source_positions]
    
    # Find mismatched positions
    mismatched = []
    for position_id in set(source_positions.keys()) & set(target_positions.keys()):
        source_position = source_positions[position_id]
        target_position = target_positions[position_id]
        
        # Compare relevant fields
        if source_position['instrument'] != target_position['instrument'] or            source_position['quantity'] != target_position['quantity'] or            source_position['value'] != target_position['value']:
            mismatched.append({
                'position_id': position_id,
                'source': source_position,
                'target': target_position
            })
    
    return {
        'total_source_positions': len(source_data),
        'total_target_positions': len(target_data),
        'missing_in_target': missing_in_target,
        'missing_in_source': missing_in_source,
        'mismatched': mismatched,
        'is_consistent': len(missing_in_target) == 0 and len(missing_in_source) == 0 and len(mismatched) == 0
    }

def reconcile_market_data(source_data: List[Dict[str, Any]], target_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconcile market data between services.
    
    Args:
        source_data: Market data from the source service
        target_data: Market data from the target service
        
    Returns:
        Reconciliation results
    """
    source_data_dict = {data['timestamp'] + data['instrument']: data for data in source_data}
    target_data_dict = {data['timestamp'] + data['instrument']: data for data in target_data}
    
    # Find missing data points
    missing_in_target = [key for key in source_data_dict if key not in target_data_dict]
    missing_in_source = [key for key in target_data_dict if key not in source_data_dict]
    
    # Find mismatched data points
    mismatched = []
    for key in set(source_data_dict.keys()) & set(target_data_dict.keys()):
        source_data_point = source_data_dict[key]
        target_data_point = target_data_dict[key]
        
        # Compare relevant fields
        if source_data_point['price'] != target_data_point['price']:
            mismatched.append({
                'key': key,
                'source': source_data_point,
                'target': target_data_point
            })
    
    return {
        'total_source_data_points': len(source_data),
        'total_target_data_points': len(target_data),
        'missing_in_target': missing_in_target,
        'missing_in_source': missing_in_source,
        'mismatched': mismatched,
        'is_consistent': len(missing_in_target) == 0 and len(missing_in_source) == 0 and len(mismatched) == 0
    }
