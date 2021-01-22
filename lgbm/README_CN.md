# DolphinDB lgbm 插件
该插件调用LightGBM库函数，可对DolphinDB内存表执行快速的回归训练、回归预测、模型保存和加载。

## 安装插件
建议直接使用我们编译好的libPluginLGBM.so,loadPlugin()函数进行加载,该动态库同时依赖于lib_lightgbm.so动态库,请将其放在同一目录内。如果需要本地编译，请执行以下步骤:

### 编译PluginLgbm动态库
在lgbm目录下执行以下代码:
```
mkdir build
cd build
cmake .. -DLIBDOLPHINDB=/path_to_dolphindb_lib/
make -j4
```
将-DLIBDOLPHINDB替换成相对应的libdolphindb.so的所在目录，编译将得到libPluginLGBM.so动态库

## 用户接口

### `lgbm::modelTrain`

#### 语法
`model=lgbm::modelTrain(X,Y,num_iteration,params)`

#### 参数
- `X`: 是一个表, 表示输入特征集;
- `Y`: 是一个向量, 表示标签列;
- `num_iteration`: 非负整值，执行一次回归训练所用的迭代次数;
- `params`: 是一个字典，用于配置相关参数，常用参数：
    - `objective`: 默认为'regression';
    - `boosting`: 默认为'gbdt', 目前插件仅支持gbdt。
    - `learning_rate`: 学习率，默认为0.1;
    - `min_data_in_leaf`: 叶子节点最少数据点个数，默认为20;
    - 更多参数请参考[官方文档](https://lightgbm.readthedocs.io/en/latest/Parameters.html)。

#### 返回
在给定数据集上对lgbm模型进行回归训练，返回一个训练好的lgbm模型。

### `lgbm::modeltrainKFold`

#### 语法
`model=lgbm::modeltrainKFold(X,Y,num_iteration,params,k)`

#### 参数
- `X`: 是一个表, 表示输入特征集;
- `Y`: 是一个向量, 表示标签列;
- `num_iteration`: 非负整值，执行一次回归训练所用的迭代次数;
- `params`: 是一个字典，用于配置相关参数，常用参数：
    - `objective`: 默认为'regression';
    - `boosting`: 默认为'gbdt', 目前插件仅支持gbdt。
    - `learning_rate`: 学习率，默认为0.1;
    - `min_data_in_leaf`: 叶子节点最少数据点个数，默认为20;
    - 更多参数请参考[官方文档](https://lightgbm.readthedocs.io/en/latest/Parameters.html)。
- `k`: 非负整值，1到10之间，对模型进行k折交叉训练。

#### 返回
在给定数据集上对lgbm模型进行k折交叉训练，获得一个较强的学习模型，返回训练得到的模型。

### `lgbm::modelPredict`

#### 语法
`Y=lgbm::modelPredict(X,model)`

#### 参数
- `X`: 是一个表，表示输入特征集;
- `model`: 一个训练好的lgbm模型。

#### 返回
对给定特征集进行回归预测，返回预测值，是一个向量。

### `lgbm::modelSave`

#### 语法
`lgbm::modelSave(path,model)`

#### 参数
- `path`: 保存路径，格式: `XXX/model.txt`;
- `model`: 要保存的lgbm模型。


#### 返回
将模型保存为文件，返回void。

### `lgbm::modelLoad`

#### 用法
`model=lgbm::modelLoad(path)`

#### 参数
- `path`: 读取路径，格式为: `xxx/model.txt`。

#### 返回
从文件中加载模型，返回得到的lgbm模型。

### `lgbm::fitGoodness`

#### 用法
`score=lgbm::fitGoodness(real,pred)`

#### 参数
- `real`: 标签列实际值;
- `pred`: 标签列预测值;

#### 返回
计算实际值和预测值间的拟合优度，返回0到1之间的双精度数。

### `lgbm::modelTrainDS`

#### 语法
`model=lgbm::modelTrainDS(ds, yColName,xColNames,params)`

#### 参数
- `ds`: 数据源，可以由sqlDS生成;
- `yColName`: 是一个字符串, 表示标签列的名字;
- `xColNames`: 是一个字符串向量, 表示特征列的名字;
- `params`: 是一个字典，用于配置相关参数，常用参数：
    - `objective`: 默认为'regression';
    - `boosting`: 默认为'gbdt', 目前分布式仅支持gbdt。
    - `num_iterations`: 执行一次回归训练所用的迭代次数，默认为100；
    - `learning_rate`: 学习率，默认为0.1;
    - `min_data_in_leaf`: 叶子节点最少数据点个数，默认为20;
    - 更多参数请参考[官方文档](https://lightgbm.readthedocs.io/en/latest/Parameters.html)。
   
    **注意**：params中部分参数必须按要求设定，或者不设定，系统会自动设定：tree_learner: "data"; pre_partition: "true" ；boost_from_average: "false" ；ignore_column: "";  weight_column: ""; group_column: "";

#### 返回
在给定数据集上对lgbm模型进行回归训练，返回一个训练好的lgbm模型。

## 使用范例

```
loadPlugin("path/to/Pluginlgbm.txt")
use lgbm;

//创建训练集
x1=rand(10,100)
x2=rand(10,100)
Y=2*x1+3*x2
X=table(x1,x2)

//设置模型参数
num_iteration=500
params = {task:"train",min_data_in_leaf:"5"}

//模型训练
model=lgbm::modelTrain(X,Y,num_iteration,params);

//创建测试集
x1=rand(10,10)
x2=rand(10,10)
X=table(x1,x2)
Y=2*x1+3*x2

//模型预测
pred=lgbm::modelPredict(X,model);

//评估预测效果
fitscore=lgbm::fitGoodness(Y,pred);

//模型保存
path="/home/zkluo/dolphindb_workspace/demo/scripts/model.txt";
lgbm::modelSave(path,model)

//模型加载
model1=lgbm::modelLoad(path);

//k折交叉训练
x1=rand(10,1000)
x2=rand(20,1000)
x3=rand(5,1000)
x4=rand(100,1000)
Y=2*x1+3*x2+0.5*x3+0.2*x4
X=table(x1,x2,x3,x4)
num_iteration=500
params = {task:"train",min_data_in_leaf:"5"}
k=10
model=lgbm::modelTrainKFold(X,Y,num_iteration,params,k);

//使用dfs表作为训练集
dbName="dfs://lgbm"
if (existsDatabase(dbName)){
	dropDatabase(dbName)
	}
db=database("dfs://lgbm",RANGE,0 500000 1000000 1500000 2000000)

t = table(0..1999999 as id, rand(10.0,2000000) as x0, rand(10.0,2000000) as x1,rand(10.0,2000000) as x2,rand(10.0,2000000) as x3,rand(10.0,2000000) as x4), rand(10.0,2000000) as y)
pt=db.createPartitionedTable(t,`pt,`id).append!(t)

ds = sqlDS(<select * from pt>)
xColNames =`x0`x1`x2`x3`x4
yColName = `y
params =dict (string, any)
params[`num_iterations]=100
model = lgbm::modelTrainDS(ds, yColName,xColNames,params)
```












