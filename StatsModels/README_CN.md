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
loadPlugin("/path/to/PluginSignal.txt");
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
返回一个table,包含adftest的统计信息，有7个字段:{"adfStat","pValue","usedLag","nobs","1%","5%","10%"}

## 2. statsmodels::kpss
对时间序列作kpsstest,返回统计信息

### 语法
```
statsmodels::adfuller(X,[regression="c"],[nLags="legacy"])
```

### 参数
- X: vector\<double\>, 输入的时间序列;
- regression: string, kpsstest的null hypothesis, 可设置为{"c","ct"}, 默认为"c";
- nLags: string或者int, 用于设定lag值，当为string时，可取值为{"auto","legacy"},其中"auto"采用Hobijn(1998)数据独立方法计算lag值，"legacy"采用整值(12*(n/100)^(1/4));当为int值，则用该值作为lag值;

### 返回值
返回一个table,包含adftest的统计信息，有7个字段:{"kpssStat","pValue","nLags","10%","5%","2.5%","1%"}

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
```




