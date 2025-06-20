#!/usr/bin/env python3
"""
Test script for the KVTrans Engine pybind11 extension
"""

import sys
import os
import numpy as np

# Add current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import kvtrans_engine
    print("✓ Successfully imported kvtrans_engine")
except ImportError as e:
    print(f"✗ Failed to import kvtrans_engine: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)

def test_engine_creation():
    """Test creating Engine instances"""
    print("\n=== Testing Engine Creation ===")
    
    # Create engines with different configurations
    try:
        engine1 = kvtrans_engine.Engine("eth0", 2, 4, 12345)
        print(f"✓ Created engine: {engine1}")
        
        engine2 = kvtrans_engine.Engine("ib0", 4, 8, 54321)
        print(f"✓ Created engine: {engine2}")
        
        return engine1, engine2
    except Exception as e:
        print(f"✗ Failed to create engine: {e}")
        return None, None

def test_connection_operations(engine):
    """Test connection operations"""
    print(f"\n=== Testing Connection Operations ===")
    
    if engine is None:
        print("✗ No engine available for testing")
        return None, None
    
    # Test connection
    print("Testing connection...")
    success, conn_id = engine.connect("127.0.0.1", 8080)
    if success:
        print(f"✓ Connected successfully with conn_id: {conn_id}")
    else:
        print("✗ Connection failed")
        return None, None
    
    # Test accept (this will simulate an incoming connection)
    print("Testing accept...")
    success, ip_addr, port, accept_conn_id = engine.accept()
    if success:
        print(f"✓ Accepted connection from {ip_addr}:{port} with conn_id: {accept_conn_id}")
    else:
        print("✗ Accept failed")
        accept_conn_id = None
    
    return conn_id, accept_conn_id

def test_kv_operations(engine, conn_id):
    """Test key-value operations"""
    print(f"\n=== Testing KV Operations ===")
    
    if engine is None or conn_id is None:
        print("✗ No engine or connection available for testing")
        return None
    
    # Create test data
    test_data = b"Hello, KVTrans Engine! This is test data for the key-value store."
    buffer = bytearray(test_data)
    
    print(f"Test data: {test_data.decode()}")
    print(f"Buffer size: {len(buffer)} bytes")
    
    # Register KV
    print("Registering KV...")
    success, kv_id = engine.reg_kv(conn_id, buffer)
    if success:
        print(f"✓ KV registered with kv_id: {kv_id}")
    else:
        print("✗ KV registration failed")
        return None
    
    # Send KV
    print("Sending KV...")
    send_data = b"Data to be sent via RDMA"
    send_buffer = bytearray(send_data)
    success = engine.send_kv(kv_id, send_buffer)
    if success:
        print(f"✓ KV sent successfully: {send_data.decode()}")
    else:
        print("✗ KV send failed")
    
    # Receive KV
    print("Receiving KV...")
    success, received_data = engine.recv_kv(kv_id, 1024)
    if success:
        print(f"✓ KV received successfully: {received_data.decode()}")
    else:
        print("✗ KV receive failed")
    
    return kv_id

def test_numpy_integration(engine, conn_id):
    """Test integration with NumPy arrays"""
    print(f"\n=== Testing NumPy Integration ===")
    
    if engine is None or conn_id is None:
        print("✗ No engine or connection available for testing")
        return
    
    # Create NumPy arrays
    print("Creating NumPy arrays...")
    array1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    array2 = np.random.rand(100).astype(np.float32)
    
    print(f"Array 1: {array1}")
    print(f"Array 2 shape: {array2.shape}, dtype: {array2.dtype}")
    
    # Register arrays as KV
    success1, kv_id1 = engine.reg_kv(conn_id, array1)
    success2, kv_id2 = engine.reg_kv(conn_id, array2)
    
    if success1 and success2:
        print(f"✓ Arrays registered with kv_ids: {kv_id1}, {kv_id2}")
    else:
        print("✗ Array registration failed")
        return
    
    # Send arrays
    success1 = engine.send_kv(kv_id1, array1)
    success2 = engine.send_kv(kv_id2, array2)
    
    if success1 and success2:
        print("✓ Arrays sent successfully")
    else:
        print("✗ Array sending failed")

def test_buffer_utilities():
    """Test buffer utility functions"""
    print(f"\n=== Testing Buffer Utilities ===")
    
    # Test buffer creation
    print("Creating test buffer...")
    # Note: create_buffer might not work as expected due to memory management
    # buffer = kvtrans_engine.create_buffer(100, 42)
    # print(f"✓ Created buffer with size: 100")
    
    # Test buffer to string conversion
    test_data = b"Test string data"
    buffer = bytearray(test_data)
    try:
        string_result = kvtrans_engine.buffer_to_string(buffer)
        print(f"✓ Buffer to string: '{string_result}'")
    except Exception as e:
        print(f"✗ Buffer to string failed: {e}")

def test_error_handling(engine):
    """Test error handling"""
    print(f"\n=== Testing Error Handling ===")
    
    if engine is None:
        print("✗ No engine available for testing")
        return
    
    # Test invalid connection ID
    print("Testing invalid connection ID...")
    success, kv_id = engine.reg_kv(999, b"test data")
    if not success:
        print("✓ Correctly handled invalid connection ID")
    else:
        print("✗ Should have failed with invalid connection ID")
    
    # Test invalid KV ID
    print("Testing invalid KV ID...")
    success = engine.send_kv(999, b"test data")
    if not success:
        print("✓ Correctly handled invalid KV ID")
    else:
        print("✗ Should have failed with invalid KV ID")

def test_module_info():
    """Test module information"""
    print("\n=== Module Information ===")
    print(f"Module docstring: {kvtrans_engine.__doc__}")
    
    available_classes = [name for name in dir(kvtrans_engine) if not name.startswith('_')]
    print(f"Available classes/functions: {available_classes}")
    
    # Test Engine class methods
    if 'Engine' in available_classes:
        engine = kvtrans_engine.Engine("test", 1, 1, 1234)
        engine_methods = [method for method in dir(engine) if not method.startswith('_')]
        print(f"Engine methods: {engine_methods}")

def main():
    """Run all tests"""
    print("Running KVTrans Engine tests...")
    
    test_module_info()
    
    # Create test engines
    engine1, engine2 = test_engine_creation()
    
    if engine1:
        # Test with first engine
        conn_id, accept_conn_id = test_connection_operations(engine1)
        
        if conn_id:
            kv_id = test_kv_operations(engine1, conn_id)
            test_numpy_integration(engine1, conn_id)
        
        test_error_handling(engine1)
    
    test_buffer_utilities()
    
    print("\n=== All KVTrans Engine tests completed! ===")

if __name__ == "__main__":
    main() 