# UNIX 查看库依赖
`ldd XXX`
# git 克隆远程仓库指定某个分支
`git clone -b <branch> <add>`

# OpenMp 提供了对并行计算的高层抽象，降低了并行编程的难度和复杂度。

## 基本语法规则:
`#pragma omp [指令] [子句[子句]...]`

## 常用指令:
1. `parallel for` 用在for循环之前，表示这段代码将被多个线程并行执行;
```
#pragma omp parallel for 
{
   for(...){...}
}
```
2. `parallel sections` 用在可能会被并行执行的代码段之前，通常配合section来使用:
```
#pragma omp parallel sections
{
   #pragma omp section
   {...}
} 
3. `schedule(static)` 配置omp调度方式,默认即为static，既当循环次数为n,线程个数为t时，则每个线程平均迭代n/t次
4. `num_threads()` 设置线程数
5. `omp_get_thread_num()` 获取线程id
6. `OMP_NUM_THREADS()` 获取可用线程数
7. `private(x)` 将变量x声明为私有，每个线程持有该变量的一个副本，互不影响
8. `firstprivate(x)` 将变量x声明为私有，且继承原变量的值
9. `lastprivate(x)` 将变量x声明为私有，在退出时返回给原来的变量
10. `shared(x)` 将X声明为共享变量,注意这将显示的产生临界区
11. `omp_lock_t x` 声明互斥锁x
12. `omp_init_lock(x)` 初始化互斥锁x
13. `omp_set_lock(x)` 获取互斥锁x
14. `omp_unset_lock(x)` 释放互斥锁x
15. `omp_destory_lock(x)` 销毁互斥锁x
16. `omp_get_num_procs()` 获得最大线程数量 

# valgrid查看内存泄漏、perf分析程序性能
