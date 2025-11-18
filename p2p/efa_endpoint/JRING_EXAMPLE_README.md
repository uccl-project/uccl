# jring 环形缓冲区使用示例

## 简介

这个示例程序演示了如何使用 `/fsx/ubuntu/shuangma/uccl/include/util/jring.h` 中定义的高性能无锁环形缓冲区。

jring 是一个基于 DPDK/FreeBSD 的环形缓冲区实现，支持：
- 单生产者/多生产者模式 (SP/MP)
- 单消费者/多消费者模式 (SC/MC)
- 批量操作 (bulk) - 全部成功或全部失败
- 突发操作 (burst) - 尽可能多地入队/出队
- 可变元素大小
- 无锁、线程安全的实现

## 文件说明

- `jring_example.cpp` - 示例程序源代码
- `Makefile.jring_example` - 编译 Makefile
- `JRING_EXAMPLE_README.md` - 本文档

## 编译和运行

### 编译

```bash
make -f Makefile.jring_example
```

### 运行

```bash
./jring_example
```

或者直接编译并运行：

```bash
make -f Makefile.jring_example run
```

### 清理

```bash
make -f Makefile.jring_example clean
```

## 示例说明

### 示例 1: 单生产者单消费者 (SPSC)

演示最简单的使用场景：
- 初始化一个 ring buffer
- 使用 `jring_sp_enqueue_bulk()` 批量入队消息
- 使用 `jring_sc_dequeue_bulk()` 批量出队消息
- 查询 ring 的状态 (count, free_count, empty, full)

**适用场景**: 单线程生产者和单线程消费者，性能最高

### 示例 2: 突发操作 (Burst)

演示 burst 模式的使用：
- 尝试入队比 ring 容量更多的元素
- burst 模式会尽可能多地入队，而不是全部或全部失败
- 使用 `jring_sp_enqueue_burst()` 和 `jring_sc_dequeue_burst()`

**适用场景**: 不需要严格保证批量操作原子性的场景

### 示例 3: 多生产者多消费者 (MPMC)

演示多线程场景：
- 2 个生产者线程同时入队
- 2 个消费者线程同时出队
- 使用 `jring_mp_enqueue_bulk()` 和 `jring_mc_dequeue_bulk()`
- 展示了线程安全的无锁操作

**适用场景**: 多线程环境下的高性能消息传递

## 关键 API 说明

### 初始化

```c
// 计算需要的内存大小
size_t ring_mem_size = jring_get_buf_ring_size(element_size, ring_size);

// 分配内存 (必须对齐到 cache line)
struct jring* ring = (struct jring*)aligned_alloc(CACHE_LINE_SIZE, ring_mem_size);

// 初始化 ring
// mp: 1=多生产者, 0=单生产者
// mc: 1=多消费者, 0=单消费者
jring_init(ring, ring_size, element_size, mp, mc);
```

### 入队操作

- `jring_sp_enqueue_bulk()` - 单生产者批量入队
- `jring_mp_enqueue_bulk()` - 多生产者批量入队
- `jring_sp_enqueue_burst()` - 单生产者突发入队
- `jring_mp_enqueue_burst()` - 多生产者突发入队
- `jring_enqueue_bulk()` - 自动检测模式的批量入队
- `jring_enqueue_burst()` - 自动检测模式的突发入队

### 出队操作

- `jring_sc_dequeue_bulk()` - 单消费者批量出队
- `jring_mc_dequeue_bulk()` - 多消费者批量出队
- `jring_sc_dequeue_burst()` - 单消费者突发出队
- `jring_mc_dequeue_burst()` - 多消费者突发出队
- `jring_dequeue_bulk()` - 自动检测模式的批量出队
- `jring_dequeue_burst()` - 自动检测模式的突发出队

### 状态查询

- `jring_count()` - 当前元素数量
- `jring_free_count()` - 剩余空闲空间
- `jring_empty()` - 是否为空
- `jring_full()` - 是否已满

## 性能提示

1. **元素大小**: 元素大小必须是 4 的倍数，建议使用 16 字节的倍数以获得最佳性能
2. **Ring 大小**: 必须是 2 的幂次方 (2, 4, 8, 16, 32, ...)
3. **内存对齐**: Ring 结构体必须对齐到 cache line (64 字节)
4. **选择正确的模式**:
   - SP/SC 模式性能最高，但只能用于单线程
   - MP/MC 模式支持多线程，性能略低但仍然很高
5. **Bulk vs Burst**:
   - Bulk 模式要求全部成功，适合需要原子性的场景
   - Burst 模式尽力而为，适合对部分失败容忍的场景

## 注意事项

- Ring 的实际容量是 `size - 1`，因为需要区分满和空的状态
- 必须使用 `aligned_alloc()` 或类似方法确保内存对齐
- 在多线程环境下，确保使用正确的 MP/MC 变体
- 释放 ring 时使用 `free(ring)` 即可

## 参考资料

- DPDK ring library: https://doc.dpdk.org/guides/prog_guide/ring_lib.html
- FreeBSD buf_ring: https://www.freebsd.org/cgi/man.cgi?query=buf_ring
