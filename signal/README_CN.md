# DolphinDB Signal Plugin

DolphinDB的signal插件对四个基础的信号处理函数（离散正弦变换、离散余弦变换、离散小波变换、反离散小波变换）进行了封装。用户可以在DolphinDB数据库软件中加载该插件以使用这四个函数进行信号处理。
新增离散余弦变换的分布式版本。

## 构建

该插件使用CMake编译（version >= 3.10)

```
mkdir build
cd build
cmake ..
make
```
## 插件加载

编译生成 libPluginSignal.so 之后，通过以下脚本加载插件：

```
loadPlugin("/path/to/PluginSignal.txt");
```

# API
## 1. signal::dct
对离散信号作离散余弦变换，返回变换序列
### 语法

```
signal::dct(X)
```

### 参数
- X: 输入的离散信号序列，应当是一个int或double类型的vector。

### 返回值

返回变换后的序列向量，与输入向量等长，元素类型为double。

## 2. signal::dst
对离散信号作离散正弦变换，返回变换序列
### 语法

```
signal::dst(X)
```

### 参数
- X: 输入的离散信号序列，应当是一个int或double类型的vector。

### 返回值

返回变换后的序列向量，与输入向量等长，元素类型为double。

## 3. signal::dwt
对离散信号作一维离散小波变换，返回由变换序列组成的table
### 语法

```
signal::dwt(X)
```

### 参数
- X: 输入的离散信号序列，应当是一个int或double类型的vector。

### 返回值

返回变换序列组成的table，包含两个字段：cA，cD。cA对应变换后的近似部分序列，cD对应变换后的细节部分序列，两个序列等长。

## 4. signal::idwt
对一维离散小波变换得到的两个序列作逆变换，返回得到的信号序列
### 语法

```
signal::idwt(X,Y)
```

### 参数
- X: 输入的近似部分序列（cA），应当是一个int或double类型的vector。
- Y: 输入的细节部分序列（cD），应当是一个int或double类型的vector。

### 返回值

返回逆变换得到的信号序列。

## 1. signal::dctParallel
离散余弦变换的分布式版本，对离散信号作离散余弦变换，返回变换序列
### 语法

```
signal::dct(ds)
```

### 参数
- ds: 输入的数据源元组，包含若干个分区，分布在若干个控制节点中。

### 返回值

返回变换后的序列向量，与输入向量等长，元素类型为double。

# 示例

## 例1 dct离散余弦变换

```
path="/path/to/PluginSignal.txt"
loadPlugin(path)
X = [1,2,3,4]
```
对信号作离散余弦变换：

```
> signal::dct(X)
[5,-2.23044235292127,-2.411540739456585E-7,-0.15851240125353]
```

## 例2 dst离散正弦变换

```
path="/path/to/Pluginignal.txt"
loadPlugin(path)
X = [1,2,3,4]
```
对信号作离散正弦变换：

```
> signal::dst(X)
[15.388417979126893,-6.88190937668141,3.632712081813623,-1.624597646358306]
```

## 例3 dwt离散小波变换

```
path="/path/to/PluginSignal.txt"
loadPlugin(path)
X = [1,2,3]
```
对信号作离散小波变换：

```
> signal::dwt(X)
cA                cD                
----------------- ------------------
2.121320343559643 -0.707106781186548
4.949747468305834 -0.707106781186548
```

## 例4 idwt反离散小波变换

```
path="/path/to/PluginSignal.txt"
loadPlugin(path)
X = [2.121320343559643,4.949747468305834]
Y = [-0.707106781186548,-0.707106781186548]
```
对序列作反离散小波变换：

```
> signal::dwt(x,y)
[1,2,3.000000000000001,4.000000000000001]
```
## 例5 dctParallel离散余弦变换分布式版本
```
f1=0..9999
f2=1..10000
t=table(f1,f2)
db = database("dfs://rangedb_data", RANGE, 0 5000 10000)
signaldata = db.createPartitionedTable(t, "signaldata", "f1")
signaldata.append!(t)
signaldata=loadTable(db,"signaldata")
ds=sqlDS(<select * from signaldata >)
loadPlugin("/path/to/PluginSignal.txt")
use signal
signal::dctParallel(ds);
```

