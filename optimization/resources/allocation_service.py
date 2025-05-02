\
"""
Implements intelligent resource allocation strategies during high load periods.
"""
import time
import random

class ResourceMonitor:
    """Simulates monitoring system resources like CPU, memory, network."""
    def get_load(self):
        """Returns a simulated system load indicator (e.g., 0.0 to 1.0)."""
        # In a real system, this would query OS metrics, container stats, etc.
        load = random.uniform(0.1, 0.95) # Simulate varying load
        print(f"Current simulated system load: {load:.2f}")
        return load

    def get_available_resources(self):
        """Returns simulated available resources."""
        # Example: {'cpu_cores': 4, 'memory_gb': 8, 'network_bw_mbps': 100}
        available = {
            'cpu_cores': random.randint(1, 8),
            'memory_gb': random.randint(4, 16),
            'network_bw_mbps': random.randint(50, 500)
        }
        print(f"Current simulated available resources: {available}")
        return available

class IntelligentAllocator:
    def __init__(self, high_load_threshold=0.8):
        """
        Initializes the allocator with a threshold to trigger intelligent allocation.
        """
        self.monitor = ResourceMonitor()
        self.high_load_threshold = high_load_threshold
        self.current_allocations = {} # Track allocated resources {task_id: resources}
        self.task_queue = [] # Queue for tasks when resources are limited
        self.task_priorities = {} # Store task priorities {task_id: priority}
        print(f"Initialized IntelligentAllocator with high load threshold: {self.high_load_threshold}")

    def request_resources(self, task_id, requested_resources, priority=0):
        """
        Processes a resource request, potentially adjusting based on load.

        Args:
            task_id (str): Identifier for the task requesting resources.
            requested_resources (dict): Resources requested (e.g., {'cpu': 1, 'memory': 2}).
            priority (int): Priority of the task (higher value = higher priority, default=0)

        Returns:
            dict or None: Allocated resources, or None if allocation fails.
        """
        print(f"\nReceived resource request for task '{task_id}' with priority {priority}: {requested_resources}")
        # Store task priority
        self.task_priorities[task_id] = priority
        
        current_load = self.monitor.get_load()
        available_resources = self.monitor.get_available_resources() # Check current availability

        if current_load >= self.high_load_threshold:
            print(f"High load detected ({current_load:.2f} >= {self.high_load_threshold}). Applying intelligent allocation.")
            # Strategy 1: Prioritization - prioritize high priority tasks
            if self._should_prioritize(task_id, priority):
                print(f"Task '{task_id}' has high priority {priority}. Attempting to allocate resources by preemption.")
                allocated_resources = self._try_preemption_allocation(requested_resources, available_resources, priority)
                if allocated_resources:
                    print(f"Successfully allocated resources for high-priority task '{task_id}' via preemption")
                    self.current_allocations[task_id] = allocated_resources
                    return allocated_resources
            
            # Strategy 2: Scaling down request (Simple example)
            allocated_resources = self._scale_down_request(requested_resources, available_resources)
            
            # Strategy 3: Queueing if can't allocate now
            if not allocated_resources:
                print(f"Cannot allocate resources now. Adding task '{task_id}' to queue.")
                self._add_to_queue(task_id, requested_resources, priority)
                return None
        else:
            print("Normal load detected. Attempting to allocate requested resources.")
            allocated_resources = self._check_and_allocate(requested_resources, available_resources)
            
            # If can't allocate under normal load, add to queue
            if not allocated_resources:
                print(f"Cannot allocate resources under normal load. Adding task '{task_id}' to queue.")
                self._add_to_queue(task_id, requested_resources, priority)
                return None

        if allocated_resources:
            print(f"Allocated resources for task '{task_id}': {allocated_resources}")
            self.current_allocations[task_id] = allocated_resources
        else:
            print(f"Failed to allocate resources for task '{task_id}'.")

        return allocated_resources
        
    def _should_prioritize(self, task_id, priority):
        """Determines if a task should be prioritized based on its priority level."""
        # Simple example: tasks with priority > 5 get prioritization treatment
        return priority > 5
        
    def _try_preemption_allocation(self, requested_resources, available_resources, priority):
        """
        Attempts to allocate resources for a high-priority task by preempting lower priority tasks.
        """
        # Check what resources we need beyond what's currently available
        needed_cpu = max(0, requested_resources.get('cpu', 0) - available_resources.get('cpu_cores', 0))
        needed_memory = max(0, requested_resources.get('memory', 0) - available_resources.get('memory_gb', 0))
        needed_network = max(0, requested_resources.get('network', 0) - available_resources.get('network_bw_mbps', 0))
        
        if needed_cpu <= 0 and needed_memory <= 0 and needed_network <= 0:
            # We have enough resources available without preemption
            return self._check_and_allocate(requested_resources, available_resources)
            
        print(f"Need to find additional resources via preemption: CPU={needed_cpu}, Memory={needed_memory}, Network={needed_network}")
        
        # Find lower priority tasks that can be preempted
        preemptable_tasks = []
        for task_id, resources in self.current_allocations.items():
            task_priority = self.task_priorities.get(task_id, 0)
            if task_priority < priority:
                preemptable_tasks.append((task_id, resources, task_priority))
                
        # Sort by priority (lowest first - these get preempted first)
        preemptable_tasks.sort(key=lambda x: x[2])
        
        # Try to get enough resources by preemption
        cpu_reclaimed = memory_reclaimed = network_reclaimed = 0
        tasks_to_preempt = []
        
        for task_id, resources, task_priority in preemptable_tasks:
            tasks_to_preempt.append(task_id)
            cpu_reclaimed += resources.get('cpu', 0)
            memory_reclaimed += resources.get('memory', 0)
            network_reclaimed += resources.get('network', 0)
            
            # Check if we have enough resources now
            if (cpu_reclaimed >= needed_cpu and 
                memory_reclaimed >= needed_memory and 
                network_reclaimed >= needed_network):
                break
                
        # If we found enough resources, preempt tasks and allocate
        if (cpu_reclaimed >= needed_cpu and 
            memory_reclaimed >= needed_memory and 
            network_reclaimed >= needed_network):
            
            print(f"Found enough resources by preempting tasks: {tasks_to_preempt}")
            
            # Preempt tasks - add them back to the queue with their priority
            for task_id in tasks_to_preempt:
                resources = self.current_allocations[task_id]
                priority = self.task_priorities.get(task_id, 0)
                print(f"Preempting task '{task_id}' with priority {priority}")
                self._add_to_queue(task_id, resources, priority)
                del self.current_allocations[task_id]
                
            # Recalculate available resources after preemption
            new_available = {
                'cpu_cores': available_resources.get('cpu_cores', 0) + cpu_reclaimed,
                'memory_gb': available_resources.get('memory_gb', 0) + memory_reclaimed,
                'network_bw_mbps': available_resources.get('network_bw_mbps', 0) + network_reclaimed
            }
            
            # Now allocate with the new available resources
            return self._check_and_allocate(requested_resources, new_available)
        else:
            print("Could not find enough resources even with preemption")
            return None
            
    def _add_to_queue(self, task_id, requested_resources, priority):
        """Adds a task to the queue when resources cannot be allocated immediately."""
        queue_item = {
            'task_id': task_id,
            'resources': requested_resources,
            'priority': priority,
            'timestamp': time.time()
        }
        self.task_queue.append(queue_item)
        print(f"Added task '{task_id}' to queue. Queue length: {len(self.task_queue)}")
        
    def process_queue(self):
        """
        Processes the task queue, attempting to allocate resources to queued tasks.
        This should be called periodically to check if resources have become available.
        """
        if not self.task_queue:
            print("Task queue is empty. Nothing to process.")
            return
            
        print(f"Processing resource allocation queue. {len(self.task_queue)} tasks waiting.")
        
        # Sort queue by priority (highest first) and then by timestamp (oldest first)
        self.task_queue.sort(key=lambda x: (-x['priority'], x['timestamp']))
        
        # Get current available resources
        available_resources = self.monitor.get_available_resources()
        current_load = self.monitor.get_load()
        
        # Keep track of remaining tasks after this process
        remaining_queue = []
        
        for task in self.task_queue:
            task_id = task['task_id']
            requested_resources = task['resources']
            priority = task['priority']
            
            print(f"Attempting to allocate resources for queued task '{task_id}' with priority {priority}")
            
            # Try to allocate based on current system load
            if current_load >= self.high_load_threshold:
                allocated = self._scale_down_request(requested_resources, available_resources)
            else:
                allocated = self._check_and_allocate(requested_resources, available_resources)
                
            if allocated:
                print(f"Successfully allocated resources for queued task '{task_id}': {allocated}")
                self.current_allocations[task_id] = allocated
                
                # Update available resources
                if 'cpu' in allocated and 'cpu_cores' in available_resources:
                    available_resources['cpu_cores'] -= allocated['cpu']
                if 'memory' in allocated and 'memory_gb' in available_resources:
                    available_resources['memory_gb'] -= allocated['memory']
                if 'network' in allocated and 'network_bw_mbps' in available_resources:
                    available_resources['network_bw_mbps'] -= allocated['network']
            else:
                print(f"Still cannot allocate resources for task '{task_id}'. Keeping in queue.")
                remaining_queue.append(task)
                
        # Update the queue with remaining tasks
        self.task_queue = remaining_queue
        print(f"Queue processing complete. {len(self.task_queue)} tasks still waiting.")


# Example Usage (can be removed or expanded)
if __name__ == '__main__':
    allocator = IntelligentAllocator(high_load_threshold=0.7) # Set threshold for testing

    request1 = {'cpu': 2, 'memory': 4, 'min_cpu': 1, 'min_memory': 2} # Task 1 needs 2 CPU, 4GB RAM
    request2 = {'cpu': 6, 'memory': 10} # Task 2 needs 6 CPU, 10GB RAM

    # Simulate a few requests
    for i in range(5):
        print(f"\n--- Allocation Cycle {i+1} ---")
        task_id_1 = f"task_1_{i}"
        task_id_2 = f"task_2_{i}"

        allocator.request_resources(task_id_1, request1)
        time.sleep(0.1) # Simulate time passing
        allocator.request_resources(task_id_2, request2)
        time.sleep(0.1)

        # Simulate task completion
        if i % 2 == 0 and task_id_1 in allocator.current_allocations:
             allocator.release_resources(task_id_1)

    print("\nFinal Allocations:", allocator.current_allocations)
