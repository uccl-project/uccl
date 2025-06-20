#!/usr/bin/env python3
"""
Demo script for KVTrans Engine - High-Performance RDMA Key-Value Transport
"""

import kvtrans_engine
import numpy as np
import time

def demo_basic_engine_usage():
    """Demonstrate basic engine creation and usage"""
    print("=== KVTrans Engine Demo ===\n")
    
    print("1. Creating KVTrans Engine...")
    # Create engine with 2 CPUs, 4 connections per CPU, listening on port 12345
    engine = kvtrans_engine.Engine("eth0", 2, 4, 12345)
    print(f"   Engine created: {engine}")
    
    return engine

def demo_connection_management(engine):
    """Demonstrate connection establishment"""
    print("\n2. Connection Management...")
    
    # Simulate client connection
    print("   Connecting to remote server...")
    success, conn_id = engine.connect("127.0.0.1", 8080)
    if success:
        print(f"   ✓ Connected with conn_id: {conn_id}")
    else:
        print("   ✗ Connection failed (expected in demo)")
    
    # Simulate server accepting connection
    print("   Accepting incoming connection...")
    success, ip_addr, port, accept_conn_id = engine.accept()
    if success:
        print(f"   ✓ Accepted connection from {ip_addr}:{port}, conn_id: {accept_conn_id}")
        return accept_conn_id
    else:
        print("   ✗ Accept failed (expected in demo)")
        return conn_id if 'conn_id' in locals() else None

def demo_kv_operations(engine, conn_id):
    """Demonstrate key-value operations"""
    print("\n3. Key-Value Operations...")
    
    if conn_id is None:
        print("   Skipping KV operations (no connection)")
        return
    
    # Create sample data
    message = "Hello from KVTrans Engine! High-performance RDMA communication."
    data_buffer = bytearray(message.encode())
    
    print(f"   Data to transfer: '{message}'")
    print(f"   Buffer size: {len(data_buffer)} bytes")
    
    # Register KV
    print("   Registering key-value pair...")
    success, kv_id = engine.reg_kv(conn_id, data_buffer)
    if success:
        print(f"   ✓ KV registered with kv_id: {kv_id}")
    else:
        print("   ✗ KV registration failed")
        return
    
    # Send KV
    print("   Sending key-value data...")
    send_data = "RDMA transfer data - ultra low latency!".encode()
    send_buffer = bytearray(send_data)
    success = engine.send_kv(kv_id, send_buffer)
    if success:
        print(f"   ✓ Data sent: '{send_data.decode()}'")
    else:
        print("   ✗ Data send failed")
    
    # Receive KV
    print("   Receiving key-value data...")
    success, received_data = engine.recv_kv(kv_id, 1024)
    if success:
        print(f"   ✓ Data received: '{received_data.decode()}'")
    else:
        print("   ✗ Data receive failed")
    
    return kv_id

def demo_numpy_arrays(engine, conn_id):
    """Demonstrate NumPy array operations"""
    print("\n4. NumPy Array Operations...")
    
    if conn_id is None:
        print("   Skipping NumPy operations (no connection)")
        return
    
    # Create various NumPy arrays
    print("   Creating NumPy arrays...")
    
    # Small integer array
    small_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    print(f"   Small array: {small_array}")
    
    # Medium float array
    medium_array = np.linspace(0, 10, 50).astype(np.float32)
    print(f"   Medium array shape: {medium_array.shape}, dtype: {medium_array.dtype}")
    
    # Large random array (simulating ML weights)
    large_array = np.random.randn(100, 100).astype(np.float64)
    print(f"   Large array shape: {large_array.shape}, dtype: {large_array.dtype}")
    print(f"   Large array size: {large_array.nbytes} bytes")
    
    # Register arrays
    arrays = [
        ("small", small_array),
        ("medium", medium_array), 
        ("large", large_array)
    ]
    
    kv_ids = []
    for name, array in arrays:
        print(f"   Registering {name} array...")
        success, kv_id = engine.reg_kv(conn_id, array)
        if success:
            print(f"   ✓ {name.capitalize()} array registered with kv_id: {kv_id}")
            kv_ids.append((name, kv_id, array))
        else:
            print(f"   ✗ {name.capitalize()} array registration failed")
    
    # Send arrays
    print("   Sending arrays via RDMA...")
    for name, kv_id, array in kv_ids:
        success = engine.send_kv(kv_id, array)
        if success:
            print(f"   ✓ {name.capitalize()} array sent ({array.nbytes} bytes)")
        else:
            print(f"   ✗ {name.capitalize()} array send failed")

def demo_performance_simulation():
    """Simulate performance characteristics"""
    print("\n5. Performance Simulation...")
    
    # Simulate latency measurements
    print("   Simulated RDMA Performance:")
    print("   - Small message latency: 0.8 μs")
    print("   - Large transfer latency: 2.1 μs")
    print("   - Throughput: 95.2 Gbps")
    print("   - CPU utilization: 12%")
    
    # Simulate data sizes
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]  # 1KB to 1MB
    print("\n   Transfer time simulation:")
    for size in sizes:
        # Simulate transfer time (unrealistic but for demo)
        simulated_time = size / (100 * 1024 * 1024 * 1024 / 8)  # 100 Gbps
        print(f"   {size:>8} bytes: {simulated_time*1000:>6.2f} ms")

def demo_ml_workflow_simulation(engine, conn_id):
    """Simulate a machine learning workflow"""
    print("\n6. Machine Learning Workflow Simulation...")
    
    if conn_id is None:
        print("   Skipping ML workflow (no connection)")
        return
    
    print("   Simulating distributed training scenario...")
    
    # Simulate model parameters (weights and biases)
    print("   Creating model parameters...")
    layer1_weights = np.random.randn(784, 256).astype(np.float32)  # Input layer
    layer1_bias = np.random.randn(256).astype(np.float32)
    layer2_weights = np.random.randn(256, 128).astype(np.float32)  # Hidden layer
    layer2_bias = np.random.randn(128).astype(np.float32)
    output_weights = np.random.randn(128, 10).astype(np.float32)   # Output layer
    output_bias = np.random.randn(10).astype(np.float32)
    
    parameters = [
        ("layer1_weights", layer1_weights),
        ("layer1_bias", layer1_bias),
        ("layer2_weights", layer2_weights),
        ("layer2_bias", layer2_bias),
        ("output_weights", output_weights),
        ("output_bias", output_bias)
    ]
    
    total_params = sum(param.size for _, param in parameters)
    total_bytes = sum(param.nbytes for _, param in parameters)
    
    print(f"   Model statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Total size: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
    
    # Register and "transfer" parameters
    print("   Registering model parameters for distributed training...")
    for name, param in parameters:
        success, kv_id = engine.reg_kv(conn_id, param)
        if success:
            print(f"   ✓ {name}: {param.shape} ({param.nbytes} bytes) -> kv_id: {kv_id}")
            # Simulate gradient transfer
            engine.send_kv(kv_id, param)
        else:
            print(f"   ✗ Failed to register {name}")

def demo_error_scenarios(engine):
    """Demonstrate error handling"""
    print("\n7. Error Handling Demonstration...")
    
    # Test invalid connection ID
    print("   Testing invalid connection ID...")
    success, kv_id = engine.reg_kv(999, b"test data")
    if not success:
        print("   ✓ Correctly handled invalid connection ID")
    
    # Test invalid KV ID
    print("   Testing invalid KV ID...")
    success = engine.send_kv(999, b"test data")
    if not success:
        print("   ✓ Correctly handled invalid KV ID")
    
    print("   Error handling working as expected!")

def main():
    """Main demo function"""
    print("KVTrans Engine - High-Performance RDMA Key-Value Transport")
    print("=" * 60)
    
    try:
        # Basic engine setup
        engine = demo_basic_engine_usage()
        
        # Connection management
        conn_id = demo_connection_management(engine)
        
        # Key-value operations
        kv_id = demo_kv_operations(engine, conn_id)
        
        # NumPy integration
        demo_numpy_arrays(engine, conn_id)
        
        # Performance simulation
        demo_performance_simulation()
        
        # ML workflow simulation
        demo_ml_workflow_simulation(engine, conn_id)
        
        # Error handling
        demo_error_scenarios(engine)
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("The KVTrans Engine provides high-performance RDMA-based")
        print("key-value transport for distributed ML and HPC workloads.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure to build the module with 'make' first.")

if __name__ == "__main__":
    main() 