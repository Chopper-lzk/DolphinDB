# lDolphinDB插件开发深度解析
DolphinDB支持动态加载外部插件，以扩展系统功能。插件用C++编写，需要编译成".so"或".dll"共享库文件。插件开发和使用的整体流程请参考DolphinDB Plugin的[GitHub页面](https://github.com/dolphindb/DolphinDBPlugin)，开发插件的方法和注意事项请参考[DolphinDB插件开发教程](https://github.com/dolphindb/Tutorials_CN/blob/master/plugin_development_tutorial.md)。本文着重解析插件开发中的一些其他常见问题：

## 1. 创建对象

编写插件时，DolphinDB中的大部分数据对象都可以用Constant类型来表示（标量、向量、矩阵、表，等等），使用时调用ConstantSP进行操作，ConstantSP是一个经过封装的智能指针，会在变量的引用计数为0时自动释放内存，不需要用户手动释放。其他常用的派生自它的变量类型有VectorSP（向量）、TableSP（表）等。

### 1.1 创建标量

插件中创建标量可以直接使用`new`语句创建头文件ScalarImp.h中声明的类型对象，可将它赋值给一个ConstantSP；也可以使用`Util::createConstant`创建指定类型的标量，并用对应的`set`方法赋值，这种方法比较麻烦，不推荐使用。

```c++
ConstantSP i = new Int(1);                 // 相当于1i
ConstantSP i1 = Util::createConstant(DT_INT);
i1->setInt(1);                             // 也相当于1i    

ConstantSP s = new String("DolphinDB");    // 相当于"DolphinDB"
ConstantSP s1 = Util::createConstant(DT_STRING);
s1->setString("DolphinDB");                // 也相当于"DolphinDB"

ConstantSP d = new Date(2020, 11, 11);     // 相当于2020.11.11
ConstantSP d1 = Util::createConstant(DT_DATE);
//由于日期类型没有对应的set方法，而date在dolphindb中存储为int，为从1970.01.01开始经过的天数，所以可以通过换算，并用setInt进行赋值
d1->setInt(18577);                         // 也相当于2020.11.11                        												
ConstantSP voidConstant = new Void();      // 创建一个void类型变量，常用于表示空的函数参数
```

### 1.2 创建非标量

对于非标量，头文件Util.h中声明了一系列函数，用于快速创建某个类型的非标量对象。

#### 1.2.1 创建Vector

`Util::createVector`可以创建指定类型的Vector对象，需要传入数据类型和长度；`Util::createRepeatingVector`可以创建相同值的Vector对象，需要传入ConstantSP对象和长度；`Util::createIndexVector`可以创建连续的一组数的Vector对象，需要传入起始数据和长度：

```c++
VectorSP v = Util::createVector(DT_INT, 10);     // 创建一个初始长度为10的int类型向量
v->setInt(0, 60);                                // 相当于v[0] = 60

VectorSP t = Util::createVector(DT_ANY, 0);      // 创建一个初始长度为0的any类型向量（元组）
t->append(new Int(3));                           // 相当于t.append!(3)
t->get(0)->setInt(4);                            // 相当于t[0] = 4
// 这里不能用t->setInt(0, 4)，因为t是一个元组，setInt(0, 4)只对int类型的向量有效

ConstantSP tem = new Double(2.1);                // 相当于2.1    
VectorSP v1 = Util::createRepeatingVector(tem, 10); // 创建一个初始长度为10，所有数据为2.1的向量

VectorSP seq = Util::createIndexVector(5, 10);   // 相当于5..14
int seq0 = seq->getInt(0);                       // 相当于seq[0]
```

#### 1.2.2 创建Matrix

`Util::createMatrix`可以创建指定类型的Matrix对象，需要传入数据类型、列数、行数以及列容量；`Util::createDoubleMatrix`可以创建double类型的Matrix对象，需要传入列数与行数：

```c++
ConstantSP m = Util::createMatrix(DT_INT, 3, 10, 3); // 创建一个10行3列，列容量为3的int类型矩阵
ConstantSP seq = Util::createIndexVector(1, 10);     // 相当于1..10
m->setColumn(0, seq);                                // 相当于m[0]=seq

ConstantSP dm = Util::createDoubleMatrix(3, 5);      // 创建一个5行3列的double类型矩阵
```

#### 1.2.3 创建Set

`Util::creatSet`可以创建指定类型的Set对象，需要传入数据类型、SymbolBaseSP和长度。SymbolBaseSP与symbol类型相关，一个symbol类型数据被DolphinDB系统内部存储为一个整数，通过SymbolBaseSP映射到对应的字符，若不需要使用可设置为nullptr：

```c++
//创建一个初始容量为0的float类型的集合，第二个参数为SymbolBaseSP，与symbol类型相关，常设置为nullptr
SetSP s = Util::createSet(DT_FLOAT, nullptr, 0);   
s->append(new Float(2.5));                         //相当于s.append!(2.5)
```

#### 1.2.4 创建Dictionary 

`Util::createDictionary`可以创建Dictionary对象，需要传入key数据类型、SymbolBaseSP、value数据类型、SymbolBaseSP，创建之后可以调用`set`设置key和value：

```c++
VectorSP keyVec = Util::createIndexVector(1, 5);        // 相当于1..5，作为key值
VectorSP valVec = Util::createVector(DT_DOUBLE, 0, 5);  // 创建一个初始长度为0，容量为5的double类型，作为value
std::vector<double> tem{2.5, 3.3, 1.0, 6.6, 8.8};
valVec->appendDouble(tem.data(), tem.size());           // 向valVec添加数据
//创建一个key类型为int、value类型为double的字典对象，第2、第4参数为SymbolBaseSP，与symbol类型相关，常设置为nullptr
DictionarySP d = Util::createDictionary(DT_INT, nullptr, DT_DOUBLE, nullptr);
d->set(keyVec, valVec);                                 // 设置key和value
```

#### 1.2.5 创建Table

`Util::createTable`可以创建Table对象，常用以下两种方法创建Table对象：一是传入列名vector 、列类型、行数、行容量；二是传入列名vector和列向量vector。

```c++
std::vector<std::string> colNames{"col1", "col2", "col3"};         // 存放列名
std::vector<DATA_TYPE> colTypes{DT_INT, DT_BOOL, DT_STRING};       // 存放列类型
//创建一张包含三列的表，列名为col1, col2, col3, 列类型分别为int, bool, string的0行、容量为100的空表
TableSP t1= Util::createTable(colNames, colTypes, 0, 100);       

VectorSP v1 = Util::createIndexVector(0, 5);                       // 相当于0..4
VectorSP v2 = Util::createRepeatingVector(new String("Demo"), 5);  // 创建一个长度为5，所有数据为"Demo"的向量
VectorSP v3 = Util::createRepeatingVector(new Double(2.5), 5);     // 创建一个长度为5，所有数据为2.5的向量
std::vector<ConstantSP> columns;                                   // 存放列向量
//添加列向量
columns.emplace_back(v1);
columns.emplace_back(v2);
columns.emplace_back(v3);
//用上述创建的列名vector和列向量vector来创建table对象
TableSP t2 = Util::createTable(colNames, columns);           
```

#### 1.2.6 创建PartialFunction

编写插件时，有时需要固定一个函数的部分参数，可以通过`Util::createPartialFunction`创建部分应用来实现。在本例中，`myFunc1`函数需要两个整数参数，计算后返回结果；`myFunc2`函数固定了`myFunc1`第一个参数，返回只需传入一个整数参数的新函数。具体步骤为：首先使用`Util::createSystemFunction `创建一个系统函数`temFunc`，参数为`myFunc1`以及参数个数，然后使用`Util::createPartialFunction`创建一个部分应用，参数为前面创建的`temFunc`以及需要固定的参数：

```c++
ConstantSP myFunc1(Heap* heap, vector<ConstantSP>& arguments) {
    if (arguments[0]->getType() != DT_INT || arguments[1]->getType() != DT_INT) {
        throw IllegalArgumentException("myFunc1", "argument must be two integral scalars!");
    }
    int a = arguments[0]->getInt();
    int b = arguments[1]->getInt();
    int result = a * b - (a + b);
    return new Int(result);
}
FunctionDefSP myFunc2(Heap* heap, vector<ConstantSP>& arguments) {
    FunctionDefSP temFunc = Util::createSystemFunction("temFunc", myFunc1, 2, 2, false);
    ConstantSP a = new Int(10);   
    vector<ConstantSP> args = {a}; 
    return Util::createPartialFunction(temFunc, args); //固定第一个参数为10
}
```

插件描述文件命名为PluginTest.txt，内容如下：

```
test,libPluginTest.so
myFunc1,myFunc1,system,2,2,0
myFunc2,myFunc2,system,0,0,0
```

在DolphinDB中加载插件并调用函数：

```
loadPlugin("Path_to_PluginTest.txt/PluginTest.txt");  // 加载插件
re1 = test::myFunc1(10, 5);  // re1值为35
newFunc= test::myFunc2();    // 获得一个只需一个参数的新函数
re2 = newFunc(5);            // 调用新函数，re2值为35
```

## 2. 高效读写Vector和内存表中的数据

在编写插件时往往需要对Vector和内存表中的数据进行读写。DolphinDB提供了许多读写的接口函数，如果使用不当将影响读写效率，因此下面将介绍如何使用DolphinDB提供的这些接口函数实现对Vector和内存表的高效读写。

### 2.1 Vector

#### 2.1.1 读取数据

DolphinDB中的Vector有两种存储模式，一种是FastVector模式，数据存储在连续的内存块中；另一种是Big array模式，数据分段存储在多个不连续的内存中。一般而言，当Vector的大小超过1048576时，Vector会切换到Big array模式。因此大部分Vector都是Big array模式，数据存储在不连续的内存中，所以不建议直接使用数据的指针对数据进行操作，极容易出错。在编写插件时，最好使用以下介绍的几种接口对Vector中的数据进行读写。

下面以int类型为例，其他数据类型都有类似的接口：

##### 2.1.1.1 int getInt(int index)

`getInt(int index)`是直接通过下标获取对应位置的元素：

```c++
VectorSP pVec = Util::createVector(DT_INT, 100000000);
int tmp;
for(int i = 0; i < pVec->size(); ++i) {
    tmp = pVec->getInt(i) ;
}
```

##### 2.1.1.2 bool getInt(int start, int len, int* buf)

第二种方法是用`getInt(int start, int len, int* buf)`批量（如每次读取1024个）将数据复制到指定的buffer：

```c++
VectorSP pVec = Util::createVector(DT_INT, 100000000);
int tmp;
const int BUF_SIZE = 1024;
int buf[BUF_SIZE];
int start = 0;
int N = pVec->size();
while (start < N) {
    int len = std::min(N - start, BUF_SIZE);
    pVec->getInt(start, len, buf);
    for (int i = 0; i < len; ++i) {
        tmp = buf[i];
    }
    start += len;
}
```

##### 2.1.1.3 const int* getIntConst(int start, int len, int* buf)

第三种是用`getIntConst(int start, int len, int* buf)`批量（如每次读取1024个）获取只读的buffer。这个方法与前一种方法的区别在于，当指定区间的数组内存空间是连续的时候，并不复制数据到指定的缓冲区，而是直接返回内存地址，这样提升了读的效率。

```c++
VectorSP pVec = Util::createVector(DT_INT, 100000000);
int tmp;
const int BUF_SIZE = 1024;
int buf[BUF_SIZE];
int start = 0;
int N = pVec->size();
while (start < N) {
    int len = std::min(N - start, BUF_SIZE);
    const int* p = pVec->getIntConst(start, len, buf);
    for (int i = 0; i < len; ++i) {
        tmp = p[i];
    }
    start += len;
}
```

当数据量比较大时，推荐使用后两种方法，因为第一种方法每次都需要调用虚函数，开销较大，而后两种方法由于cache命中率较高、虚函数调用次数少，性能较好。下表为分别采用上述三种方法从vector中读取1亿个int并将其赋值给另一个数所花费的时间：

| 函数     | int getInt(int index) | bool getInt(int start, int len, int* buf) | const int* getIntConst(int start, int len, int* buf) |
| -------- | --------------------- | ----------------------------------------- | ---------------------------------------------------- |
| 花费时间 | 575.775 ms            | 222.789 ms                                | 121.326 ms                                           |

##### 2.1.1.4 读取String与Symbol类型Vector

String类型与Symbol类型的Vector可以通过`getString(INDEX index)`访问下标的方式来获取数据，然而每次都会调用虚函数，效率低，要实现高效读取数据与上述其他类型有所区别，所以下面将单独介绍。

###### 2.1.1.4.1  读取String类型Vector

因为String类型的数组在内存中连续存储，所以可以用`getDataArray`函数获得数组数据的指针，将其转成DolphinSring类型，然后对其进行操作：

```c++
ConstantSP readString(Heap *heap, vector<ConstantSP> &arguments) {
    if (!(arguments[0]->isVector() && arguments[0]->getType() == DT_STRING)) {
        throw IllegalArgumentException("readString", "argument must be a string vector");
    }
    VectorSP pVec = arguments[0]; //aruments[0]为需要获取数据的String类型的Vector
    size_t size = pVec->size();
    DolphinString *pDolString = (DolphinString *)pVec->getDataArray(); //获取数据指针
    for (size_t i = 0; i < size; i++) {
        std::cout << pDolString[i].getString() << std::endl; //读取数据
    }
    return new Void();
}
```

###### 2.1.1.4.2  读取Symbol类型Vector

一个symbol类型数据被DolphinDB系统内部存储为一个整数，需要通过SymbolBaseSP来映射到对应的字符，所以可以先通过上述高效读取int类型数组的方法先获取symbol，然后通过SymbolBaseSP得到对应的字符：

```c++
ConstantSP readSymbol(Heap *heap, vector<ConstantSP> &arguments) {
    if (!(arguments[0]->isVector() && arguments[0]->getType() == DT_SYMBOL)) {
        throw IllegalArgumentException("readSymbol", "argument must be a symbol vector");
    }
    VectorSP pVec = arguments[0]; //aruments[0]为需要获取数据的Symbol类型的Vector
    int buf[1024];
    int start = 0;
    int N = pVec->size();
    SymbolBaseSP pSymbol = pVec->getSymbolBase(); //获取SymbolBaseSP
    while (start < N) {
        int len = std::min(N - start, 1024);
        pVec->getInt(start, len, buf);
        for (int i = 0; i < len; ++i)
        {
            std::cout << pSymbol->getSymbol(buf[i]).getString() << std::endl; //读取数据
        }
        start += len;
    }
    return new Void();
}
```

#### 2.1.2 更新数据

同理，更新一个Vector中的数据也不建议直接使用数据的指针进行操作，建议使用下面介绍的方法更新Vector中的数据，下面同样以int类型为例，其他数据类型有类似的接口：

##### 2.1.2.1 void setInt(int index,int val)

`setInt(int index,int val)`是直接通过下标更新单个数据点：

```c++
const int size = 100000000;
VectorSP pVec = Util::createVector(DT_INT, size);
for(int i = 0; i < size; ++i) {
    pVec->setInt(i, i);
}
```

##### 2.1.2.2 bool setInt(INDEX start, int len, const int* buf)

`setInt(INDEX start, int len, const int* buf)`是批量更新len长度的连续数据点：

```c++
const int size = 100000000;
const int BUF_SIZE = 1024;
int tmp[1024];
VectorSP pVec = Util::createVector(DT_INT, size);
int start = 0;
while(start < size) {
    int len =  std::min(size - start, BUF_SIZE);
    for(int i = 0; i < len; ++i) {
        tmp[i] = i;
    }
    pVec->setInt(start, len, tmp);
    start += len;
}
```

##### 2.1.2.3 bool setData(INDEX start, int len, void* buf)

`setData(INDEX start, int len, void* buf)`也是是批量更新len长度的连续数据点，但是不负责检查数据类型：

```c++
const int size = 100000000;
const int BUF_SIZE = 1024;
int tmp[1024];
VectorSP pVec = Util::createVector(DT_INT, size);
int start = 0;
while(start < size) {
    int len =  std::min(size - start, BUF_SIZE);
    for(int i = 0; i < len; ++i) {
        tmp[i] = i;
    }
    pVec->setData(start, len, tmp);
    start += len;
}
```

##### 2.1.2.4 使用int* getIntBuffer(INDEX start, int len, int* buf)获取buffer后用setInt进行批量更新

先使用`getIntBuffer`获取向量中的一段buffer，修改数据后再用`setInt`批量更新：

```c++
const int size = 100000000;
const int BUF_SIZE = 1024;
int buf[1024];
VectorSP pVec = Util::createVector(DT_INT, size);
int start = 0;
while(start < size) {
    int len =  std::min(size - start, BUF_SIZE);
    int* p = pVec->getIntBuffer(start, len, buf);
    for(int i = 0; i < len; ++i) {
        p[i] = i;
    }
    pVec->setInt(start, len, p);
    start += len;
}
```

上面介绍的四种方法，`setInt(int index,int val)`更新单个数据会反复调用虚函数，效率最低；而`setData`不检查类型，这两种方法都不推荐。建议使用`setInt(INDEX start, int len, const int* buf)`批量进行更新，而在`setInt`之前使用`getIntBuffer`先获取一段buff，修改数据之后然后再调用`setInt`能提高效率。这是因为`getIntBuffer`方法一般情况下会直接返回内部地址，只有在区间[start, start + len)跨越vector的内存交界处时，才会将数据拷贝至用户传入的buffer。而setInt方法会判断传入的buffer地址是否为内部存储的地址，如果是则直接返回，否则进行内存拷贝。所以先调用`getIntBuffer`再调用setInt会减少内存拷贝的次数，从而提高效率。下表为分别采取上述四个方法更新vector中1亿个数据所花费的时间：

| 方法     | void setInt(int index,int val) | bool setInt(INDEX start, int len, const int* buf) | bool setData(INDEX start, int len, void* buf) | 使用getIntBuffer获取buffer后用setInt进行批量更新 |
| -------- | ------------------------------ | ------------------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| 花费时间 | 445.679 ms                     | 226.483 ms                                        | 225.877 ms                                    | 126.297 ms                                       |

### 2.2 Table

对于Table类型可用`getColumn`函数获取某一列的指针，然后使用[2.1小节](#21-vector)介绍的方法对得到的整列数据进行处理，实现高效读取：

```c++
TableSP ddbTbl = input;
for (size_t i = 0; i < columnSize; ++i) {
    ConstantSP col = input->getColumn(i);
// ...
}
```

向Table中添加数据，可以使用`bool append(vector<ConstantSP>& values, INDEX& insertedRows, string& errMsg)`方法，如果插入成功，返回true，并向insertedRows中写入插入的行数；否则返回false，并在errMsg中写入出错信息，并不会抛出异常，需要用户处理出错的情况。本例中传入两个Table对象，将第二个Table的数据添加到第一个Table中，两个Table的列数需要相同，否则会出错：

```c++
ConstantSP append(Heap *heap, vector<ConstantSP> &arguments) {
    if (!(arguments[0]->isTable() && arguments[1]->isTable())) {
        throw IllegalArgumentException("append", "arguments need two tables");
    }
    TableSP t1 = arguments[0];
    TableSP t2 = arguments[1];
    size_t columnSize = t2->columns();
    std::vector<ConstantSP> dataToAppend;
    for (size_t i = 0; i < columnSize; i++) {
        dataToAppend.emplace_back(t2->getColumn(i));
    }
    INDEX insertedRows;
    std::string errMsg;
    bool success = t1->append(dataToAppend, insertedRows, errMsg);
    if (!success)
        std::cerr << errMsg << std::endl;
    return new Void();
}
```

## 3. 插件中创建线程

编写插件时，如果需要创建线程，不建议使用`pthread_create`等函数直接创建线程，这样创建的线程没有初始化DolphinDB中的一些随机数，在执行线程函数时可能会出错。建议使用`new`语句创建在头文件Concurrent.h中声明的Thread对象，它需要传入一个RunnableSP对象，所以需要用户实现一个Runnable对象的派生类，并完成纯虚函数`void run()`的实现，用`void run()`来执行线程函数，然后将Runnable派生类对象的SmartPointer传给Thread。本例中DemoRun为Runnable的派生类，`void run()`函数读取一个整型数组的数据并打印，`createThread`函数创建了一个线程通过`start()`函数来执行DemoRun的`run()`函数，并`join()`回收该线程：

```c++
class DemoRun : public Runnable {
public:
    DemoRun(ConstantSP data) : data_(data) {}
    ~DemoRun() {}
    void run() override {
        size_t size = data_->size();
        for (size_t i = 0; i < size; ++i) {
            std::cout << data_->getInt(i) << std::endl;
        }
    }
private:
    ConstantSP data_;
};

ConstantSP createThread(Heap *heap, vector<ConstantSP> &arguments) {
    if (!(arguments[0]->isVector() && arguments[0]->getType() == DT_INT)) {
        throw IllegalArgumentException("createThread", "argument must be an integral vector");
    }
    SmartPointer<DemoRun> demoRun = new DemoRun(arguments[0]);
    ThreadSP thread = new Thread(demoRun);
    if (!thread->isStarted()) {
        thread->start();
    }
    thread->join();
    return new Void();
}
```

由于`join()`函数会阻塞等待子线程的退出，如果子线程需要长时间后台运行，则上述方法不合理。子线程运行时需要保证ThreadSP对象仍然存在，因此可以新建一个类，将ThreadSP作为其成员函数，通过`new`语句创建这个类，并用其创建子线程并执行线程函数，然后通过返回`Util::createResource`创建的对象来管理这个类的资源释放，因此需要实现一个回调函数`demoOnClose`，当需要回收资源时会自动调用这个回调函数释放资源，用户也可以通过其他函数来手动释放资源：

```c++
class DemoRun2 : public Runnable {
public:
    DemoRun2(ConstantSP data) : data_(data) {}
    ~DemoRun2() {}
    void run() override {
        size_t size = data_->size();
        for (size_t i = 0; i < size; ++i) {
            std::cout << data_->getInt(i) << std::endl;
            Util::sleep(2000);
        }
    }
private:
    ConstantSP data_; 
};

class Demo {
public:
    Demo() {}
    ~Demo() {}
    void createAndRun(ConstantSP data) {
        SmartPointer<DemoRun2> demoRun = new DemoRun2(data);
        thread_ = new Thread(demoRun);
        if (!thread_->isStarted()) {
            thread_->start();
        }
    }
private:
    ThreadSP thread_;
};

static void demoOnClose(Heap* heap, vector<ConstantSP>& args) {
    Demo* pDemo = (Demo*)(args[0]->getLong());
    if(pDemo != nullptr) {
        delete pDemo;
        args[0]->setLong(0);
    }
}

ConstantSP createThread2(Heap *heap, vector<ConstantSP> &arguments) {
    if (!(arguments[0]->isVector() && arguments[0]->getType() == DT_INT)) {
        throw IllegalArgumentException("createThread", "argument must be an integral vector");
    }
    Demo * pDemo = new Demo();
    pDemo->createAndRun(arguments[0]);
    FunctionDefSP onClose(Util::createSystemProcedure("demo onClose()", demoOnClose, 1, 1));
    return Util::createResource(reinterpret_cast<long long>(pDemo), "Demo", onClose, heap->currentSession());
}
```

此外介绍下会话的概念：会话是一个容器，它具有唯一的ID并存储许多已经定义的对象，例如局部变量、共享变量等。创建新会话的方式有多种，如启动命令行窗口、XDB连接、 GUI连接或Web URL连接。 会话中的所有变量对于其他会话是不可见的，除非使用语句share在会话之间显式共享变量， 目前DolphinDB仅支持表共享。创建新线程时，线程函数如果需要用到会话的一些资源（比如调用内置函数），则需要创建一个新的会话，否则当前会话关闭后（比如在GUI里调用，然后关闭GUI），如果线程函数仍在执行，会导致程序崩溃。本例中

```c++
class appendTable : public Runnable {
public:
    appendTable(Heap *heap, TableSP table, ConstantSP handle, Client* client) 
    : heap_(heap), table_(table), handle_(handle), client_(client) {
        session_ = heap->currentSession()->copy();
        //session_->setUser(heap->currentSession()->getUser());
        //session_->setOutput(new DummyOutput);
    }
    ~appendTable() {}
    void append() {
        if(handle_->isTable()) {
            std::vector<ConstantSP> dataToAppend = {table_, handle_};
            session_->getFunctionDef("append!")->call(session_->getHeap().get(), dataToAppend);
        }
        else{
            std::vector<ConstantSP> dataToAppend = {table_};
            ((FunctionDefSP)handle_)->call(session_->getHeap().get(), dataToAppend);
        }
    }
    void run() override {
        while(true) {
            append();
            Util::sleep(5000);
        }
    }
private:
    SessionSP session_;
    Client * client_;
    ConstantSP handle_;
    TableSP table_;
    bool flag_ = true;
};

class Client {
public:
    Client() {}
    ~Client() {}
    void createAndRun(Heap *heap, ConstantSP table, ConstantSP handle) {
        session_ = heap->currentSession()->copy();
        session_->setUser(heap->currentSession()->getUser());
        session_->setOutput(new DummyOutput);
        SmartPointer<appendTable> append = new appendTable(heap, table, handle, this);
        thread_ = new Thread(append);
        if (!thread_->isStarted()) {
            thread_->start();
        }
    }
    SessionSP getSession() {
        return session_;
    }
private:
    ThreadSP thread_;
};

static void clientOnClose(Heap* heap, vector<ConstantSP>& args) {
    Client* pClient = (Client*)(args[0]->getLong());
    if(pClient != nullptr) {
        delete pClient;
        args[0]->setLong(0);
    }
}

ConstantSP createThread3(Heap *heap, vector<ConstantSP> &arguments) {
    if (!(arguments[0]->isTable() && arguments[1]->isTable())) {
        throw IllegalArgumentException("append", "arguments need two tables");
    }
    Client * pClient = new Client();
    pClient->createAndRun(heap, arguments[0], arguments[1]);
    FunctionDefSP onClose(Util::createSystemProcedure("client onClose()", clientOnClose, 1, 1));
    return Util::createResource(reinterpret_cast<long long>(pClient), "client", onClose, heap->currentSession());
}
```





## 4. 用户权限

首先介绍下会话的概念：会话是一个容器，它具有唯一的ID并存储许多已经定义的对象，例如局部变量、共享变量等。创建新会话的方式有多种，如启动命令行窗口、XDB连接、 GUI连接或Web URL连接。 会话中的所有变量对于其他会话是不可见的，除非使用语句[share](https://www.dolphindb.cn/cn/help/Share.html)在会话之间显式共享变量， 目前DolphinDB仅支持表共享。创建线程时，如果新线程会用到会话的一些资源（比如访问用户权限），则此时需要创建一个新的会话，否则原会话关闭后（如关闭GUI窗口），线程如果还在执行线程函数则会出错。下面以设置用户权限为例，介绍下如何创建新会话，并设置用户权限：



执行计划作业时，用户需要具备与对象相关的权限，在编写插件获取MQTT、OPC_UA等第三方数据时，在订阅时往往需要设置用户权限。而订阅后往往需要长时间运行，所以通常需要创建新线程来处理消息，在设置权限时要用到在头文件CoreConcept.h中声明的Session对象，调用其成员函数`setUser`。而Session类可以通过Heap调用`getCurrentSesion()`获取当前会话，为了防止当前会话关闭后就收不到订阅的消息，需要调用`getCurrentSesion()->copy()`获取当前会话的拷贝，然后用这拷贝的Session通过`setUser`设置权限。而每个Session都有一个Output类负责输出，所以需要实现一个Output的派生类，然后调用拷贝的Session的`setOutput`方法设置该派生类。

下面简单介绍下设置用户权限的几个关键步骤。首先需要创建一个有个SessionSP类型的session成员的类，为了简便可以使用其构造函数负责创建线程（[第3小节](#3-插件中创建线程)介绍的方法）并负责订阅和设置用户权限；此外需要实现一个Output的派生类，实现对应的纯虚函数，在插件中Output通常不需要处理输出，所以可以实现一个如下所示的DummyOutput：

```c++
class DemoRun:public Runnable {
public:
    DemoRun() { ... }
    ~DemoRun() { ... }
    void run() override {
    //...
    }
}

class DummyOutput: public Output{
public:
    virtual bool timeElapsed(long long nanoSeconds){return true;}
    virtual bool write(const ConstantSP& obj){return true;}
    virtual bool message(const string& msg){return true;}
    virtual void enableIntermediateMessage(bool enabled) {}
    virtual IO_ERR done(){return OK;}
    virtual IO_ERR done(const string& errMsg){return OK;}
    virtual bool start(){return true;}
    virtual bool start(const string& message){return true;}
    virtual IO_ERR writeReady(){return OK;}
    virtual ~DummyOutput(){}
    virtual OUTPUT_TYPE getOutputType() const {return STDOUT;}
    virtual void close() {}
    virtual void setWindow(INDEX index,INDEX size){};
    virtual IO_ERR flush() {return OK;}
};

class Client {
public:
    Client(Heap *heap, ...) {
        //创建线程
        if(run == nullptr) {
            run = new DemoRun();
            thread = new Thread(run);
        }
        
    	//订阅
        //...
        
        //设置权限
        session = heap->currentSession()->copy();
        session->setUser(heap->currentSession()->getUser());
        session->setOutput(new DummyOutput());
    }
    ~Client() { ... }
    //...
private:
    DemoRun *run = nullptr;
    ThreadSP thread;
    SessionSP session;
    //...
};
```

然后在需要设置用户权限的插件接口函数（通常是订阅函数）中创建上面实现的类：

```c++
ConstantSP subscribe(Heap* heap, vector<ConstantSP>& arguments) {
    //参数检验等
    //...
    std::unique_ptr<Client> conn(new Client(heap, ...));
    //...
}
```

完整的过程可参考[mqtt插件](https://github.com/dolphindb/DolphinDBPlugin/tree/master/mqtt) 或者[opcua插件](https://github.com/dolphindb/DolphinDBPlugin/tree/master/opcua)。

