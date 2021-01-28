/*
 * signal.cpp
 *
 * Created on: Dec 1.2020
 *     Author: zkluo
 *
 */

#include "signal.h"
#include "math.h"
#include "Util.h"
#include <vector>
#include <string>
#include <omp.h>
#include <ScalarImp.h>

#define PI 3.1415926
#define  DOLPHIN_NULL -1.7976931348623157e+308
static void dwt_get(int,int,vector<double>& ,vector<double>&, vector<double>&);
static void idwt_get(int,int,vector<double>& ,vector<double>&, vector<double>&,omp_lock_t&);
static bool checkNull(vector<double>&);

static bool checkNull(vector<double>& x){
    for(int idx=0;idx<x.size();idx++){
        if(x[idx]==DOLPHIN_NULL)
            return false;
    }
    return true;
}
//离散余弦变换(DCT-II)
ConstantSP dct(const ConstantSP& a, const ConstantSP& b){
    string funName="dct";
    string syntax="Usage: "+funName+"(x)";
    if(!(a->isArray()&&a->isNumber()&&(a->getCategory()==INTEGRAL || a->getCategory()==FLOATING)&&a->size()>0))
        throw IllegalArgumentException("dct", "The argument should be a nonempty integrial or floating one dimension vector.");
    int size=a->size();
    vector<double> xn(size,0);    //存储输入的离散信号序列x(n)
    vector<double> xk(size,0);    //存储计算出的离散余弦变换序列X(k)
    a->getDouble(0,size,&xn[0]);
    if(!checkNull(xn))
        throw IllegalArgumentException(funName,syntax+" Invalid input, check NULL contained.");
#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for(int k=0;k<size;k++){
        double data=0;
        double ak=k==0?sqrt(1.0/size):sqrt(2.0/size);
        double base_cos=cos(PI*2*k/(2*size));
        double base_sin=sin(PI*2*k/(2*size));
        double last_cos,last_sin,cur_cos,cur_sin;
        for(int j=0;j<size;j++) {
            cur_cos = j==0?cos(PI*k/(2*size)):last_cos * base_cos-last_sin * base_sin; //cos(kj)=cos((k-1)j+j)=cos((k-1)j)*cos(j)-sin((k-1)j)*sin(j)
            cur_sin = j==0?sin(PI*k/(2*size)):last_sin * base_cos+last_cos * base_sin;//sin(kj)=sin((k-1)j+j)=sin((k-1)j)cosj+cos((k-1)j)*sinj
            last_cos=cur_cos;
            last_sin=cur_sin;
            data += xn[j] * cur_cos;
        }
        xk[k]=ak*data;
    }
    VectorSP res=Util::createVector(DT_DOUBLE,size);
    res->setDouble(0,size,&xk[0]);
    return res;
}
ConstantSP dctMap(Heap *heap, vector<ConstantSP> &args){
    TableSP table=args[0];
    int size=args[1]->getInt();
    vector<double> xn(table->rows(),0);
    vector<double> xk(size,0);
    vector<int> index_j(table->rows(),0);
    table->getColumn(0)->getInt(0,table->rows(),&index_j[0]);
    table->getColumn(1)->getDouble(0,table->rows(),&xn[0]);
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for(int idx=0;idx<index_j.size();idx++) {
        for (int k = 0; k < size; k++) {
            double ak = k == 0 ? sqrt(1.0 / size) : sqrt(2.0 / size);
            omp_set_lock(&lock);
            xk[k] += xn[idx] * cos(PI * k * (2 * index_j[idx] + 1) / (2 * size)) * ak;
            omp_unset_lock(&lock);
        }
    }
    ConstantSP result=Util::createVector(DT_DOUBLE,size);
    result->setDouble(0,size,&xk[0]);
    return result;
}
ConstantSP dctNumMap(Heap *heap, vector<ConstantSP> &args){
    TableSP t=args[0];
    int size=t->rows();
    ConstantSP res=Util::createConstant(DT_INT);
    res->setInt(size);
    return res;
}
ConstantSP dctReduce(const ConstantSP &mapRes1, const ConstantSP &mapRes2){
    vector<double> xk_1(mapRes1->size(),0);
    vector<double> xk_2(mapRes2->size(),0);
    mapRes1->getDouble(0,mapRes1->size(),&xk_1[0]);
    mapRes2->getDouble(0,mapRes2->size(),&xk_2[0]);
#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for(int i=0;i<xk_1.size();i++)
        xk_1[i]+=xk_2[i];
    ConstantSP result=Util::createVector(DT_DOUBLE,xk_1.size());
    result->setDouble(0,xk_1.size(),&xk_1[0]);
    return result;
}
ConstantSP dctNumReduce(const ConstantSP &mapRes1, const ConstantSP &mapRes2){
    int x1=mapRes1->getInt();
    int x2=mapRes2->getInt();
    int size=x1+x2;
    ConstantSP res=Util::createConstant(DT_INT);
    res->setInt(size);
    return res;
}
ConstantSP dctParallel(Heap* heap,vector<ConstantSP>& args){
    ConstantSP ds=args[0];

    FunctionDefSP num_mapfunc=heap->currentSession()->getFunctionDef("signal::dctNumMap");
    FunctionDefSP num_reducefunc=heap->currentSession()->getFunctionDef("signal::dctNumReduce");
    FunctionDefSP mr=heap->currentSession()->getFunctionDef("mr");
    vector<ConstantSP> num_myargs={ds,num_mapfunc,num_reducefunc};
    ConstantSP size=mr->call(heap,num_myargs);

    FunctionDefSP mapfunc=heap->currentSession()->getFunctionDef("signal::dctMap");
    vector<ConstantSP> mapwithsizearg={new Void(),size};
    FunctionDefSP mapwithsize=Util::createPartialFunction(mapfunc,mapwithsizearg);
    FunctionDefSP reducefunc=heap->currentSession()->getFunctionDef("signal::dctReduce");
    vector<ConstantSP> myargs={ds,mapwithsize,reducefunc};
    return mr->call(heap,myargs);
}
//离散正弦变换(DST-I)
ConstantSP dst(const ConstantSP& a, const ConstantSP& b){
    string funName="dst";
    string syntax="Usage: "+funName+"(x)";
    if(!(a->isArray()&&a->isNumber()&&(a->getCategory()==INTEGRAL || a->getCategory()==FLOATING)&&a->size()>0))
        throw IllegalArgumentException("dst", "The argument should be a nonempty integrial or floating one dimension vector.");
    int size=a->size();
    vector<double> xn(size,0);     //存储输入的离散信号序列x(n)
    vector<double> xk(size,0);     //存储输出的离散正弦变换序列X(k)
    a->getDouble(0,size,&xn[0]);
    if(!checkNull(xn))
        throw IllegalArgumentException(funName,syntax+" Ivalid input, check NULL contained.");
#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for(int k=0;k<size;k++){
        double ak=2;
        //double ak=sqrt(2.0/(size+1));
        double data=0;
        double base_cos=cos(PI*(k+1)/(size+1));
        double base_sin=sin(PI*(k+1)/(size+1));
        double last_cos,last_sin,cur_cos,cur_sin;
        for(int j=0;j<size;j++){
            cur_cos = j==0?base_cos:last_cos * base_cos-last_sin * base_sin; //cos(kj)=cos((k-1)j+j)=cos((k-1)j)*cos(j)-sin((k-1)j)*sin(j)
            cur_sin = j==0?base_sin:last_sin * base_cos+last_cos * base_sin;//sin(kj)=sin((k-1)j+j)=sin((k-1)j)cosj+cos((k-1)j)*sinj
            last_cos=cur_cos;
            last_sin=cur_sin;
            data+=xn[j]*cur_sin;
        }
        xk[k]=ak*data;
    }
    VectorSP res=Util::createVector(DT_DOUBLE,size);
    res->setDouble(0,size,&xk[0]);
    return res;
}

//一维离散小波变换(DWT)
ConstantSP dwt(const ConstantSP& a, const ConstantSP& b){
    string funName="dwt";
    string syntax="Usage: "+funName+"(x)";
    if(!(a->isArray()&&a->isNumber()&&(a->getCategory()==INTEGRAL || a->getCategory()==FLOATING)&&a->size()>0))
        throw IllegalArgumentException("dwt", "The argument should be a nonempty integrial or floating one dimension vector.");
    int dataLen=a->size();  //信号序列长度
    vector<double> FilterLD = {
            0.7071067811865475244008443621048490392848359376884740365883398,
            0.7071067811865475244008443621048490392848359376884740365883398
    }; //基于db1小波函数的滤波器低通序列
    vector<double> FilterHD = {
            -0.7071067811865475244008443621048490392848359376884740365883398,
            0.7071067811865475244008443621048490392848359376884740365883398
    }; //基于db1小波函数的滤波器通序列
    const int filterLen=2;    //滤波器序列长度
    int decLen=(dataLen+filterLen-1)/2; //小波变换后的序列长度
    vector<double> xn(dataLen,0);
    vector<double> cA(decLen,0);
    vector<double> cD(decLen,0);
    a->getDouble(0,dataLen,&xn[0]);
    if(!checkNull(xn))
        throw IllegalArgumentException(funName,syntax+" Invalid input, check NULL contained.");
#pragma omp sections
    {
      #pragma omp section
      {
        dwt_get(filterLen, dataLen, xn, FilterLD, cA);
      }
      #pragma omp section
      {
        dwt_get(filterLen, dataLen, xn, FilterHD, cD);
      }
    }
    VectorSP res_cA=Util::createVector(DT_DOUBLE,decLen);
    VectorSP res_cD=Util::createVector(DT_DOUBLE,decLen);
    res_cA->setDouble(0,decLen,&cA[0]);
    res_cD->setDouble(0,decLen,&cD[0]);
    vector<string> colNames={"cA", "cD"};    //cA:分解后的近似部分序列-低频部分  cD:分解后的细节部分序列-高频部分
    vector<ConstantSP> columns;
    columns.emplace_back(res_cA);
    columns.emplace_back(res_cD);
    TableSP t=Util::createTable(colNames,columns);
    return t;
}

//一维离散小波逆变换(IDWT)
ConstantSP idwt(const ConstantSP& a, const ConstantSP& b){
    string funName="idwt";
    string syntax="Usage: "+funName+"(ca,cd)";
    if(!(a->isArray()&&a->isNumber()&&(a->getCategory()==INTEGRAL || a->getCategory()==FLOATING)&&a->size()>0))
        throw IllegalArgumentException("idwt", "The argument 1 should be a nonempty integrial or floating one dimension vector.");
    if(!(b->isArray()&&a->isNumber()&&(b->getCategory()==INTEGRAL || b->getCategory()==FLOATING)&&a->size()>0))
        throw IllegalArgumentException("idwt", "The argument 2 should be a nonempty integrial or floating dimension vector.");
    if(a->size()!=b->size())
        throw IllegalArgumentException("idwt", "two arguments should have the same size.");
    vector<double> FilterLR = {
            0.7071067811865475244008443621048490392848359376884740365883398,
            0.7071067811865475244008443621048490392848359376884740365883398
    }; //基于db1小波函数的滤波器低通序列
    vector<double> FilterHR = {
            0.7071067811865475244008443621048490392848359376884740365883398,
            -0.7071067811865475244008443621048490392848359376884740365883398
    }; //基于db1小波函数的滤波器通序列
    int dataLen=a->size();
    const int filterLen=2;
    int recLen=dataLen*2;
    vector<double> cA(dataLen,0);
    vector<double> cD(dataLen,0);
    vector<double> recData(recLen,0);
    a->getDouble(0,dataLen,&cA[0]);
    b->getDouble(0,dataLen,&cD[0]);
    if((!checkNull(cA))||(!checkNull(cD)))
        throw IllegalArgumentException(funName,syntax+" Invalid input, check NULL contained.");
    omp_lock_t _lock;
    omp_init_lock(&_lock);
#pragma omp sections
    {
        #pragma omp section
        {
            idwt_get(filterLen, dataLen, cA, FilterLR, recData,_lock);
        }
        #pragma omp section
        {
            idwt_get(filterLen, dataLen, cD, FilterHR, recData,_lock);
        }
    }
    VectorSP res=Util::createVector(DT_DOUBLE,recLen);
    res->setDouble(0,recLen,&recData[0]);
    return res;
}
static void dwt_get(int decLen,int dataLen,vector<double>& input,vector<double>&Filter,vector<double>& output){
    int step=2;
    int i=step-1,idx=0;

    for(;i<decLen&&i<dataLen;i+=step,++idx){
        double sum=0;
        int j;
        for(j=0;j<=i;j++)
            sum+=Filter[j]*input[i-j];
        while(j<decLen){
            int k;
            for(k=0;k<dataLen&&j<decLen;++j,++k)
                sum+=Filter[j]*input[k];
            for(k=0;k<decLen&&j<decLen;++k,++j)
                sum+=Filter[j]*input[dataLen-1-k];
        }
        output[idx]=sum;
    }
    for(;i<dataLen;i+=step,++idx){
        double sum=0;
        for(int j=0;j<decLen;++j)
            sum+=input[i-j]*Filter[j];
        output[idx]=sum;
    }
    for(;i<decLen;i+=step,++idx){
        double sum=0;
        int j=0;
        while(i-j>=dataLen){
            int k;
            for(k=0;k<dataLen&&i-j>=dataLen;++j,++k)
                sum+=Filter[i-dataLen-j]*input[dataLen-1-k];
            for(k=0;k<dataLen&&i-j>=dataLen;++j,++k)
                sum+=Filter[i-dataLen-j]*input[k];
        }
        for(;j<=i;++j)
            sum+=Filter[j]*input[i-j];
        while(j<decLen){
            int k;
            for(k=0;k<dataLen&&j<decLen;++j,++k)
                sum+=Filter[j]*input[k];
            for(k=0;k<dataLen&&j<decLen;++k,++j)
                sum+=Filter[j]*input[dataLen-1-k];
        }
        output[idx]=sum;
    }
    for(;i<dataLen+decLen-1;i+=step,++idx){
        double sum=0;
        int j=0;
        while(i-j>=dataLen){
            int k;
            for(k=0;k<dataLen&&i-j>=dataLen;++j,++k)
                sum+=Filter[i-dataLen-j]*input[dataLen-1-k];
            for(k=0;k<dataLen&&i-j>=dataLen;++j,++k)
                sum+=Filter[i-dataLen-j]*input[k];
        }
        for(;j<decLen;++j)
            sum+=Filter[j]*input[i-j];
        output[idx]=sum;
    }
}
static void idwt_get(int recLen,int dataLen,vector<double>& input,vector<double>&Filter,vector<double>& output,omp_lock_t& _lock){
    int idx,i;
    for(idx=0,i=recLen/2-1;i<dataLen;++i,idx+=2){
        double sum_even=0;
        double sum_odd=0;
        for(int j=0;j<recLen/2;++j){
            sum_even+=Filter[j*2]*input[i-j];
            sum_odd+=Filter[j*2+1]*input[i-j];
        }
        omp_set_lock(&_lock);
        output[idx]+=sum_even;
        output[idx+1]+=sum_odd;
        omp_unset_lock(&_lock);
    }
}