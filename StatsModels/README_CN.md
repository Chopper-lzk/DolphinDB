# DolphinDB StatsModels Plugin

DolphinDB StatsModels插件对adftest和kpsstest进行了封装，用于分析时间序列。

## 编译PluginStatsmodels动态库

在StatsModels目录下执行以下代码：
```
mkdir build
cd build
cmake .. -DLIBDOLPHINDB=/path_to_dolphindb_lib/
make -j4
```

## 插件加载

编译生成libPluginStatsmodels.so,使用以下脚本加载：
```
loadPlugin("/path/to/PluginStatsmodels.txt");
```

# API

## 1. statsmodels::adfuller
对时间序列作adftest,返回统计信息

### 语法
```
statsmodels::adfuller(X,[maxLag],[regression=c])
```

### 参数
- X: vector\<double\>, 输入的时间序列;
- maxLag: int, 测试所用的最大lag值,也是最后采用的lag值，默认为12*(nobs/100)^(1/4);
- regression: string, 用于设定Constant和trend order, 可设置为{"c","ct","ctt",""}, 默认为"c";

### 返回值
返回一个table,包含adftest的统计信息，有7个字段:{"adfStats","pValues","usedLag","nobs","1%","5%","10%"}

## 2. statsmodels::kpss
对时间序列作kpsstest,返回统计信息

### 语法
```
statsmodels::kpss(X,[regression="c"],[nLags="legacy"])
```

### 参数
- X: vector\<double\>, 输入的时间序列;
- regression: string, kpsstest的null hypothesis, 可设置为{"c","ct"}, 默认为"c";
- nLags: string或者int, 用于设定lag值，当为string时，可取值为{"auto","legacy"},其中"auto"采用Hobijn(1998)数据独立方法计算lag值，"legacy"采用整值(12*(n/100)^(1/4));当为int值，则用该值作为lag值;

### 返回值
返回一个table,包含adftest的统计信息，有7个字段:{"kpssStats","pValues","nLags","10%","5%","2.5%","1%"}

## 3. statsmodels::bds
对时间序列作bdstest,返回统计信息

### 语法
```
statsmodels::bds(X,[max_dim=2],[epsilon],[distance=1.5])
```

### 参数
- X: vector\<double\>, 输入的时间序列;
- max_dim: int,最大嵌入维数,默认为2;
- epsilon: double, 用于计算相关和的阈值距离;
- distance: double, 指定测试时要使用的距离乘数，默认为1.5;

### 返回值
返回一个table,包含bds的统计信息，有2个字段:{"bdsStats","pvalues"}

## 4. statsmodels::q_stat
计算Ljung-Box Q统计量

### 语法
```
statsmodels::q_stat(X,nobs)
```

### 参数
- X: vector\<double\>, 自相关系数数组;
- nobs: int, 样本观察次数;

### 返回值
返回一个table,包含q_stat的统计信息，有2个字段:{"qStats","pValues"}

# 使用范例

```
//adftest
x=[1,3,5,7,4,6,11,4,5]
maxlag=0
regression="c"
res=statsmodels::adfuller(x,maxlag,regression)
//kpsstest
x=[1,3,5,7,4,6,11,4,5]
regression="ct"
nLags="auto"
res=statsmodels::kpss(x,regression,nLags)
//bds
x=[1,3,5,7,4,6,11,4,5]
max_dim=4
epislon=1.2
distance=2.0
res=statsmodels::bds(x,max_dim,epislon,distance)
//q_stat
x=[0.9,0.2,0.1]
nobs=10
res=statsmodels::q_stat(x,nobs)
```




