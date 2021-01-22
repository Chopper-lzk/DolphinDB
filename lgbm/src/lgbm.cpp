#include "lgbm.h"
#include <LightGBM/c_api.h>
#include <LightGBM/boosting.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/metric.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/threading.h>

#include <string>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "application/predictor.hpp"
#include <LightGBM/utils/yamc/alternate_shared_mutex.hpp>
#include <LightGBM/utils/yamc/yamc_shared_lock.hpp>
#include <stdio.h>
#include <string>
#include <fstream>
#include <thread>
#include <Util.h>
#include <ScalarImp.h>
#include "../LightGBM/src/treelearner/parallel_tree_learner.h"
using namespace LightGBM;
namespace OperatorImp{
    ConstantSP add(const ConstantSP& a, const ConstantSP& b);
}

static void BoosterUpdater(BoosterHandle, vector<double>&, int, int);
static double EvalR2(vector<double>&, vector<double>&);
static void LGBM_BoosterOnClose(Heap *heap, vector<ConstantSP> &args);
static void processNan(vector<vector<double>>&, vector<double>&);
ConstantSP constructBinMapper(LightGBM::Config& config, vector<string>& header,  const std::vector<std::vector<double>>& data, vector<int>& sampledIndex);
vector<int> getSampledIndex(LightGBM::Config& config, const std::vector<std::vector<double>>& data);
Dataset* constructDatasetWithBinMapper(vector<string>& header, ConstantSP binMappers, LightGBM::Config& config, const std::vector<std::vector<double>>& data, vector<int>& sampledIndex);

void checkParams(bool parallel, DictionarySP params){
    if(parallel){
        string funcName = "lgbm::modelTrainDS";
        string syntax = "Usage: " + funcName + "(ds, yColName, xColNames, params).";
        if(!params->getMember("pre_partition")->isNull() && (params->getMember("pre_partition")->getType() != DT_STRING || params->getMember("pre_partition")->getString() != "true" )){
            throw RuntimeException("pre_partition should be a string scalar and equal to 'true'.");
        }
        if(!params->getMember("tree_learner")->isNull() && (params->getMember("tree_learner")->getType() != DT_STRING || params->getMember("tree_learner")->getString() != "data" )){
            throw RuntimeException("tree_learner must be data.");
        }
        if(!params->getMember("boost_from_average")->isNull() && (params->getMember("boost_from_average")->getType() != DT_STRING || params->getMember("boost_from_average")->getString() != "false") ){
            throw RuntimeException("modelTrainDS not supports boost_from_average yet.");
        }
        if(!params->getMember("objective")->isNull()){
            if(params->getMember("objective")->getType() != DT_STRING)
                throw RuntimeException("objective must be string");
            string obj = params->getMember("objective")->getString();
            if(obj == "regression_l1" || obj == "quantile" || obj == "mape" || obj == "binary" || obj == "multiclass" || obj == "multiclassova")
                throw RuntimeException("modelTrainDS not supports objective " + obj + " yet.");

        }
        if(!params->getMember("boosting")->isNull()){
            if(params->getMember("boosting")->getType() != DT_STRING || params->getMember("boosting")->getString() != "gbdt"){
                throw RuntimeException("modelTrainDS only supports boosting=`gbdt");
            }
        }
        if(!params->getMember("ignore_column")->isNull()){
            throw RuntimeException("modelTrainDS not supports to set ignore_column");
        }
        if(!params->getMember("weight_column")->isNull()){
            throw RuntimeException("modelTrainDS not supports to set weight_column");
        }
        if(!params->getMember("group_column")->isNull()){
            throw RuntimeException("modelTrainDS not supports to set group_column");
        }
    }
}

ConstantSP modelTrain(Heap* heap, vector<ConstantSP>& args){
    string funcName = "lgbm::modelTrain";
    string syntax = "Usage: " + funcName + "(X, Y, num_iterations, params).";
    if(args.size() != 4)
        throw IllegalArgumentException(funcName, syntax + "Illegal argument size.");
    if(!(args[0]->isTable()))
        throw IllegalArgumentException(funcName, syntax + "X must be a table.");
    TableSP t = args[0];
    if(!t->isBasicTable())
        throw IllegalArgumentException(funcName, syntax + "X must be a basic table.");
    for(int i = 0; i < t->columns(); i++){
        if(!t->getColumn(i)->isNumber())
            throw IllegalArgumentException(funcName, syntax + "Every column of X must be of a numeric type.");
    }
    if(!(args[1]->isVector() && args[1]->isNumber()))
        throw IllegalArgumentException(funcName, syntax + "Y must be a numeric vector.");
    if(args[0]->rows() != args[1]->size())
        throw IllegalArgumentException(funcName, syntax + "the dimension doesn't match between X and Y.");
    if(!(args[2]->isScalar() && args[2]->getCategory() == INTEGRAL && args[2]->getInt() > 0))
        throw IllegalArgumentException(funcName, syntax + "num_iterations should be positive integer.");
    if(!args[3]->isDictionary())
        throw IllegalArgumentException(funcName, syntax + "params should be a dictionary.");
    DictionarySP dict = args[3];
    if(!(dict->getKeyType() == DT_STRING))
        throw IllegalArgumentException(funcName, syntax + "params should have string keys");

    //-------- end of argument validation----------

    vector<vector<double>> x(args[0]->rows(), vector<double>(args[0]->columns(), -1));
    vector<double> y(args[1]->size());
    int total_iteration = args[2]->getInt();
    string params;
    char name[256];

    for(INDEX i = 0; i < args[0]->columns(); i++){
        ConstantSP col = args[0]->getColumn(i);
        for(int j = 0; j < args[0]->rows(); j++)
            x[j][i] = col->getDouble(j);
    }
    args[1]->getDouble(0, args[1]->size(), &y[0]);

    ConstantSP keys = dict->keys();
    ConstantSP values = dict->values();
    for(int i = 0; i < keys->size(); i++){
        string k = keys->getString(i);
        string v = values->getString(i);
        params += k;
        params += "=";
        params += v;
        params += " ";
    }
    sprintf(name, "num_iterations=%d ", total_iteration);
    params += name;
    params += "label=name:y ";
    params += "header=true ";
    for(size_t i = 0; i < x.size(); i++) x[i].push_back(y[i]);
    vector<string> header;

    for(size_t i = 0; i < x[0].size() - 1; i++){
        string colName = t->getColumnName(i);
        header.push_back(colName);
    }
    header.push_back("y");

    //nan process
    processNan(x,y);

    //create dataset
    DatasetHandle traindata= nullptr;

    LGBM_DatasetCreateMain("traindata", &params[0], nullptr, &traindata, x, header);

    //create booster
    BoosterHandle booster= nullptr;
    if(LGBM_BoosterCreate(traindata, &params[0], &booster)<0)
        throw RuntimeException(LGBM_GetLastError());

    //train booster
    BoosterUpdater(booster, y, total_iteration, 0);

    //free traindataset
    if(LGBM_DatasetFree(traindata)<0)
        throw RuntimeException(LGBM_GetLastError());
    FunctionDefSP onClose(Util::createSystemProcedure("LGBM Booster onClose()", LGBM_BoosterOnClose, 1, 1));
    return Util::createResource((long long)booster, "LGBM_BOOSTER", onClose, heap->currentSession());
}

ConstantSP modelPredict(Heap* heap, vector<ConstantSP>& args){
    string funcName = "lgbm::modelPredict";
    string syntax = "Usage: " + funcName + "(X, model).";
    if(args.size() != 2)
        throw IllegalArgumentException(funcName, syntax + "Illegal argument size.");
    if(!(args[0]->isTable()))
        throw IllegalArgumentException(funcName, syntax + "X must be a table.");
    TableSP t = args[0];
    if(!t->isBasicTable())
        throw IllegalArgumentException(funcName, syntax + "X must be a basic table.");
    for(int i = 0; i < t->columns(); i++){
        if(!t->getColumn(i)->isNumber())
            throw IllegalArgumentException(funcName, syntax + "Every column of X must be of a numeric type.");
    }
    if(!(args[1]->getType() == DT_RESOURCE && args[1]->getString() == "LGBM_BOOSTER"))
        throw IllegalArgumentException(funcName, syntax + "model should be lgbm booster resource.");

    // ------------end of argument validation--------------

    vector<vector<double>> x(args[0]->rows(), vector<double>(args[0]->columns(), -1));
    vector<double> res(args[0]->rows());

    for(INDEX i = 0; i < args[0]->columns(); i++){
        ConstantSP col = args[0]->getColumn(i);
        for(int j = 0; j < args[0]->rows(); j++)
            x[j][i] = col->getDouble(j);
    }
    BoosterHandle booster = (BoosterHandle)args[1]->getLong();
    //predict
    int64_t out_len;
    for(size_t i = 0; i < x.size(); i++) {
        if (LGBM_BoosterPredictForMat(booster, &x[i][0], C_API_DTYPE_FLOAT64, 1, x[i].size(), 1,C_API_PREDICT_NORMAL, 0, -1, "", &out_len, &res[i]) < 0)
            throw RuntimeException(LGBM_GetLastError());
    }
    VectorSP out=Util::createVector(DT_DOUBLE,res.size());
    out->setDouble(0, x.size(), &res[0]);
    return out;
}

ConstantSP modelSave(Heap* heap, vector<ConstantSP>& args){
    string funcName = "lgbm::modelSave";
    string syntax = "Usage: " + funcName + "(path, model).";
    if(args.size() != 2)
        throw IllegalArgumentException(funcName, syntax + "Illegal argument size.");
    if(!(args[0]->getType() == DT_STRING && args[0]->isScalar()))
        throw IllegalArgumentException(funcName, syntax + "path should be a string.");
    string savepath = args[0]->getString();
    std::ofstream out(savepath);
    if(!out)
        throw IllegalArgumentException(funcName, syntax + "invalid path.");
    out.close();
    if(!(args[1]->getType() == DT_RESOURCE && args[1]->getString() == "LGBM_BOOSTER"))
        throw IllegalArgumentException(funcName, syntax + "model should be lgbm booster resource.");

    //---------end of argument validation
    BoosterHandle booster = (BoosterHandle) args[1]->getLong();

    if(LGBM_BoosterSaveModel(booster, 0, -1, C_API_FEATURE_IMPORTANCE_SPLIT, &savepath[0]) < 0)
    throw RuntimeException(LGBM_GetLastError());
    return Util::createConstant(DT_VOID);
}

ConstantSP modelLoad(Heap* heap, vector<ConstantSP>& args){
    string funcName = "lgbm::modelLoad";
    string syntax = "Usage: " + funcName + "(path).";
    if(args.size() != 1)
        throw IllegalArgumentException(funcName, syntax + "Illegal argument size.");
    if(!(args[0]->getType() == DT_STRING && args[0]->isScalar()))
        throw IllegalArgumentException(funcName, syntax + "path should be a string.");

    //--------end of argument validation---------

   // load model
   string modelpath = args[0]->getString();
   string modelstr;
   int num_iterations;
   BoosterHandle booster;

   if(LGBM_BoosterCreateFromModelfile(&modelpath[0], &num_iterations, &booster)<0)
       throw RuntimeException(LGBM_GetLastError());
   printf("Booster load success, current iteration: %d\n", num_iterations);
   FunctionDefSP onClose(Util::createSystemProcedure("LGBM Booster onClose()", LGBM_BoosterOnClose, 1, 1));
   return Util::createResource((long long)booster,"LGBM_BOOSTER", onClose, heap->currentSession());
}

ConstantSP modelTrainKFold(Heap* heap, vector<ConstantSP>& args){
    string funcName = "lgbm::modelTrainKFold";
    string syntax = "Usage: " + funcName + "(X, Y, num_iterations, params, k).";
    if(args.size() != 5)
        throw IllegalArgumentException(funcName, syntax + "Illegal argument size.");
    if(!(args[0]->isTable()))
        throw IllegalArgumentException(funcName, syntax + "X must be a table.");
    TableSP t = args[0];
    if(!t->isBasicTable())
        throw IllegalArgumentException(funcName, syntax + "X must be a basic table.");
    for(int i = 0; i < t->columns(); i++){
        if(!t->getColumn(i)->isNumber())
            throw IllegalArgumentException(funcName, syntax + "Every column of X must be of a numeric type.");
    }
    if(!(args[1]->isVector() && args[1]->isNumber()))
        throw IllegalArgumentException(funcName, syntax + "Y must be a numeric vector.");
    if(args[0]->rows() != args[1]->size())
        throw IllegalArgumentException(funcName, syntax + "the dimension doesn't match between X and Y.");
    if(!(args[2]->isScalar() && args[2]->getCategory() == INTEGRAL && args[2]->getInt() > 0))
        throw IllegalArgumentException(funcName, syntax + "num_iterations should be positive integer.");
    if(!args[3]->isDictionary())
        throw IllegalArgumentException(funcName, syntax + "params should be a dictionary.");
    DictionarySP dict = args[3];
    if(!(dict->getKeyType() == DT_STRING))
        throw IllegalArgumentException(funcName, syntax + "params should have string keys");
    if(!(args[4]->isScalar() && args[4]->getCategory() == INTEGRAL && args[4]->getInt() >= 1 && args[4]->getInt() <= 10))
        throw IllegalArgumentException(funcName, syntax + " k should be a positive integer in [1,10]");
    int thread_num = args[4]->getInt();

    //--------end of argument validation---------

    vector<vector<double>> x(args[0]->rows(), vector<double>(args[0]->columns(), -1));
    vector<double> y(args[1]->size());
    int total_iteration = args[2]->getInt();
    string params;
    char name[256];

    for(INDEX i = 0; i < args[0]->columns(); i++){
        ConstantSP col = args[0]->getColumn(i);
        for(int j = 0; j < args[0]->rows(); j++)
            x[j][i] = col->getDouble(j);
    }
    args[1]->getDouble(0, args[1]->size(), &y[0]);
    for(size_t i = 0; i < y.size(); i++) y[i] /= thread_num;

    ConstantSP keys = dict->keys();
    ConstantSP values = dict->values();
    for(int i = 0; i < keys->size(); i++){
        string k = keys->getString(i);
        string v = values->getString(i);
        params += k;
        params += "=";
        params += v;
        params += " ";
    }
    sprintf(name, "num_iterations=%d ", total_iteration);
    params += name;
    params += "label=name:y ";
    params += "header=true ";
    for(size_t i = 0; i < x.size(); i++) x[i].push_back(y[i]);
    vector<string> header;

    for(size_t i = 0; i < x[0].size() - 1; i++){
        string colName = t->getColumnName(i);
        header.push_back(colName);
    }
    header.push_back("y");

    //nan process
    processNan(x, y);
    int item = x.size() / thread_num;
    if(item < 1)
        throw IllegalArgumentException(funcName, syntax + "too few data for k-folding.");

    vector<vector<double>> LabelData(thread_num);
    vector<std::thread*> threadVec(thread_num);
    vector<DatasetHandle> dataVec(thread_num);
    vector<BoosterHandle> boosterVec(thread_num);

    for(int idx = 0; idx < thread_num; idx++) {
        vector<vector<double>> _data;
        for(int row = 0; row < item * thread_num; row++){
            if(!(row >= idx * item && row < (idx + 1) * item)) {
                _data.push_back(x[row]);
                LabelData[idx].push_back(y[row]);
            }
        }

        //create dataset
        LGBM_DatasetCreateMain("traindata", &params[0], nullptr, &dataVec[idx], _data,header);

        //create booster
        if (LGBM_BoosterCreate(dataVec[idx], &params[0], &boosterVec[idx]) < 0)
            throw RuntimeException(LGBM_GetLastError());

        //train booster
        threadVec[idx] = new std::thread(BoosterUpdater, boosterVec[idx], ref(LabelData[idx]), total_iteration, idx);
    }
    for(int idx = 0; idx < thread_num; idx++)
        threadVec[idx]->join();
    for(int idx = 1; idx < thread_num; idx++){
        if(LGBM_BoosterMerge(boosterVec[0], boosterVec[idx]) < 0)
            throw RuntimeException(LGBM_GetLastError());
    }

    for(int idx = 0; idx < thread_num; idx++) {
        //free booster
        if(idx != 0) {
            if (LGBM_BoosterFree(boosterVec[idx]) < 0)
                throw RuntimeException(LGBM_GetLastError());
        }
        //free traindataset
        if (LGBM_DatasetFree(dataVec[idx]) < 0)
            throw RuntimeException(LGBM_GetLastError());
        //free thread object
        delete threadVec[idx];
    }
    FunctionDefSP onClose(Util::createSystemProcedure("LGBM Booster onClose()", LGBM_BoosterOnClose, 1, 1));
    return Util::createResource((long long)boosterVec[0], "LGBM_BOOSTER", onClose, heap->currentSession());
}

ConstantSP fitGoodness(Heap* heap,vector<ConstantSP>& args){
    string funcName = "lgbm::fitgoodness";
    string syntax = "Usage: " + funcName + "(real, pred).";
    if(args.size() != 2)
        throw IllegalArgumentException(funcName, syntax + "Illegal argument size.");
    if(!(args[0]->isVector() && args[0]->isNumber()))
        throw IllegalArgumentException(funcName, syntax + "real should be a numeric vector.");
    if(!(args[1]->isVector() && args[1]->isNumber()))
        throw IllegalArgumentException(funcName, syntax + "pred should be a numeric vector.");
    if(args[0]->size() != args[1]->size())
        throw IllegalArgumentException(funcName, syntax + "two vector have different size.");

    //------------end of argument validation-------------
    VectorSP real = args[0];
    VectorSP pred = args[1];
    FunctionDefSP multiply = heap->currentSession()->getFunctionDef("multiply");
    FunctionDefSP sub = heap->currentSession()->getFunctionDef("sub");
    vector<ConstantSP> arguments = {pred, real};
    VectorSP resSub = sub->call(heap, arguments);
    double res = resSub->sum2()->getDouble();
    ConstantSP avg = real->avg();
    arguments = {real, avg};
    double tot = ((VectorSP)sub->call(heap, arguments))->sum2()->getDouble();
   
    return new Double(1.0 - res / tot);
}

static void BoosterUpdater(BoosterHandle booster, vector<double>& LabelData, int total_iteration, int thread_id){
    //train booster
    int finish_flag;
    int64_t outlen;
    double r2;
    vector<double> PredictData(LabelData.size(), -1);
    for(int i = 0; i < total_iteration; i++) {
        if (LGBM_BoosterUpdateOneIter(booster, &finish_flag) < 0)
            throw RuntimeException(LGBM_GetLastError());

       if(LGBM_BoosterGetPredict(booster, 0, &outlen, &PredictData[0]) < 0)
           throw RuntimeException(LGBM_GetLastError());
        r2=EvalR2(LabelData, PredictData);
        printf("[thread %u] [Iteration %d] r2: %f\n", thread_id, i, r2);
    }
    printf("[thread %u] LGBM_BoosterUpdate finished, total_iteration: %d\n", thread_id, total_iteration);
}

static double EvalR2(vector<double>& real,vector<double>& pre){
    double avg = 0;
    double res = 0;
    double tot = 0;
    for(size_t i = 0; i < real.size(); i++)
        avg += real[i] / real.size();
    for(size_t i = 0; i < real.size(); i++){
        res += (pre[i] - real[i]) * (pre[i] - real[i]);
        tot += (real[i] - avg) * (real[i] - avg);
    }
    return 1 - res / tot;
}

static void LGBM_BoosterOnClose(Heap *heap, vector<ConstantSP> &args) {
    BoosterHandle booster = (BoosterHandle) args[0]->getLong();
    if (nullptr != booster) {
        if(LGBM_BoosterFree(booster) < 0)
            throw RuntimeException(LGBM_GetLastError());
        args[0]->setLong(0);
    }
}
static void processNan(vector<vector<double>>&x, vector<double>& y){
    for(size_t i = 0; i < x.size();){
        bool flag = true;
        for(size_t j = 0; j < x[i].size(); j++){
            if(x[i][j] == DBL_NMIN){
                x.erase(x.begin() + i);
                y.erase(y.begin() + i);
                flag = false;
                break;
            }
        }
        if(flag)  i++;
    }
}

ConstantSP modelTrainDS(Heap* heap, vector<ConstantSP>& args){
    string funcName = "lgbm::modelTrainDS";
    string syntax = "Usage: " + funcName + "(ds, yColName, xColNames, params).";
    if(!args[0]->isTuple())
        throw IllegalArgumentException(funcName, syntax + "ds must be a list of data souces.");
    ConstantSP ds = args[0];
    ConstantSP yColName = args[1];
    ConstantSP xColNames = args[2];
    if(!yColName->isScalar() || yColName->getType() != DT_STRING){
        throw IllegalArgumentException(funcName, syntax + "yColName should be a string scalar.");
    }
    if(!xColNames->isVector() || yColName->getType() != DT_STRING){
        throw IllegalArgumentException(funcName, syntax + "xColNames should be a string array.");
    }
    if (!args[3]->isDictionary()) {
        throw IllegalArgumentException(funcName, syntax + "params must be a dictionary with string keys.");
    }
    DictionarySP params = args[3]->getValue();
    if (params->getKeyType() != DT_STRING) {
        throw IllegalArgumentException(funcName, syntax + "params must be a dictionary with string keys.");
    }
    checkParams(true, params);
    if(!params->getMember("num_machines")->isNull()){
        params->set("num_machines", new Int(2));
    }
    if(!params->getMember("num_machine")->isNull()){
        params->set("num_machine", new Int(2));
    }
    params->set("num_machines", new Int(2));
    params->set("pre_partition", new String("true"));
    params->set("tree_learner", new String("data"));
    params->set("boost_from_average", new String("false"));
    
    
    
    FunctionDefSP imrDS = heap->currentSession()->getFunctionDef("imr");
    FunctionDefSP _mapFunc = Util::createSystemFunction("lgbmMap", &lgbmMap, 6, 6, false);
    ConstantSP initValue = Util::createVector(DT_ANY, 2);
    initValue->set(0, new Short(0));
    initValue->set(1, new Void());
    vector<ConstantSP> mapArgs{new Void(), new Void(), new Void(), yColName, xColNames, params};
    FunctionDefSP mapFunc = Util::createPartialFunction(_mapFunc, mapArgs);

    FunctionDefSP reduceFunc = Util::createOperatorFunction("lgbmReduce", &lgbmReduce, 2, 2, false);
    FunctionDefSP finalFunc = Util::createSystemFunction("lgbmFinal", &lgbmFinal, 2, 2, false);
    FunctionDefSP termFunc = Util::createOperatorFunction("lgbmTerminate", &lgbmTerminate, 2, 2, false);
    vector<ConstantSP> imrArgs{ds, initValue, mapFunc, reduceFunc, finalFunc, termFunc, new Bool(true)};
    ConstantSP imrRes = imrDS->call(heap, imrArgs);
    BoosterHandle booster= nullptr;
    int currentIter;
    if(LGBM_BoosterLoadModelFromString(imrRes->getString().c_str(), &currentIter, &booster)<0)
        throw RuntimeException(LGBM_GetLastError());
    FunctionDefSP onClose(Util::createSystemProcedure("LGBM Booster onClose()", LGBM_BoosterOnClose, 1, 1));
    return Util::createResource((long long)booster, "LGBM_BOOSTER", onClose, heap->currentSession());
}

ConstantSP convertFeatureHistogram(LightGBM::FeatureHistogram* featureHistogram){
    int size = featureHistogram->SizeOfHistgram() / sizeof(double);
    ConstantSP vec = Util::createVector(DT_DOUBLE, size);
    double *pbuf = (double *)vec->getDataArray();
    memcpy(pbuf, featureHistogram->RawData(), size);
    return vec;
}

ConstantSP lgbmMap(Heap* heap, vector<ConstantSP>& args){
    TableSP table = args[0];
    ConstantSP initValue = args[1];
    ConstantSP carryOver = args[2];
    ConstantSP yColName = args[3];
    ConstantSP xColNames = args[4];
    ConstantSP param = args[5];
    int isFinish;
    ConstantSP mapResult = Util::createVector(DT_ANY,2);
    ConstantSP mapRet = Util::createVector(DT_ANY,2);
    if(carryOver->isNull()){
        mapResult->set(1, new Long(0));
    }
    else{
        mapResult->set(1, carryOver);
    }
    mapResult->set(0, mapRet);
    
    if(carryOver->isNull() || carryOver->isVector()){
        INDEX rows = table->rows();
        INDEX cols = xColNames->size();
        vector<vector<double>> x(rows, vector<double>(cols + 1, -1));
        string params;
        vector<string> header;
        string colName;
        for(INDEX i = 0; i < cols; i++){
            colName = xColNames->getString(i);
            header.push_back(colName);
            int colIndex = table->getColumnIndex(colName);
            if (colIndex == -1)
                throw RuntimeException("Table does not contain column: " + xColNames->getString(i));
            ConstantSP col = table->getColumn(colIndex);
            for(INDEX j = 0; j < rows; j++){
                x[j][i] = col->getDouble(j);
            }
        }
        colName = yColName->getString();
        header.push_back(colName);
        int colIndex = table->getColumnIndex(colName);
        if (colIndex == -1)
            throw RuntimeException("Table does not contain column: " + yColName->getString());
        ConstantSP col = table->getColumn(colIndex);
        for(INDEX j = 0; j < rows; j++){
            x[j][cols] = col->getDouble(j);
        }
        ConstantSP keys = param->keys();
        ConstantSP values = param->values();
        for(int i = 0; i < keys->size(); i++){
            string k = keys->getString(i);
            string v = values->getString(i);
            params += k;
            params += "=";
            params += v;
            params += " ";
        }
        params += "label=name:" + colName;
        params += " header=true ";

        //create dataset
        auto parameters = Config::Str2Map(params.c_str());
        Config config;
        config.Set(parameters);
        if (config.num_threads > 0) {
            omp_set_num_threads(config.num_threads);
        }
        if(carryOver->isNull()){
            vector<int> sampleIndex = getSampledIndex(config, x);
            ConstantSP binMappers = constructBinMapper(config, header, x, sampleIndex);
            mapRet->set(0, new Char(0));// task = 0 gather binMapper
            mapRet->set(1, binMappers);
            ConstantSP indices = Util::createVector(DT_INDEX, sampleIndex.size(), sampleIndex.size(), true);
            memcpy((int *)indices->getDataArray(), sampleIndex.data(), sampleIndex.size());
            mapResult->set(1, indices);

        }
        else{
            ConstantSP indices = carryOver;
            vector<int> sampleIndex(indices->size());
            memcpy(sampleIndex.data(), (int *)indices->getDataArray(), sampleIndex.size());
            Dataset* traindata = constructDatasetWithBinMapper(header, initValue->get(1), config, x, sampleIndex);
            LightGBM::DDBApplication* booster = new LightGBM::DDBApplication(params.c_str(), (LightGBM::Dataset*)traindata);
            ConstantSP handle = new Long((long long)booster);
            isFinish = booster->TrainOneIter();
            if(isFinish){
                ConstantSP dumpModel = new String(booster->getModel());
                mapRet->set(0, new Char(5));// task = 5 dumpModel
                mapRet->set(1, dumpModel);
                mapResult->set(1, new Void());
                delete booster;
            }
            else{
                LightGBM::TrainStatus status = booster->getTrainStatus();
                if(status == LightGBM::TrainStatus::sumSplitInfo){
                    ConstantSP res = Util::createVector(DT_ANY, 3);
                    int num_data_in_leaf;
                    double sum_gradients;
                    double sum_hessians;
                    booster->getSplitInfo(&num_data_in_leaf, &sum_gradients, &sum_hessians);
                    res->set(0, new Int(num_data_in_leaf));
                    res->set(1, new Double(sum_gradients));
                    res->set(2, new Double(sum_hessians));
                    mapRet->set(0, new Char(1)); // task = 1 sum SplitInfo
                    mapRet->set(1, res);
                    mapResult->set(1, handle);
                }
                else{
                    throw RuntimeException("Error train status");
                }
            }
        }
    }
    else{
        LightGBM::DDBApplication* booster = (LightGBM::DDBApplication* )(carryOver->getLong());
        LightGBM::TrainStatus status = booster->getTrainStatus();
        if(status == LightGBM::TrainStatus::sumSplitInfo){
            int num_data_in_leaf;
            double sum_gradients;
            double sum_hessians;
            num_data_in_leaf = initValue->get(1)->get(0)->getInt();
            sum_gradients = initValue->get(1)->get(1)->getDouble();
            sum_hessians = initValue->get(1)->get(2)->getDouble();

            booster->setSplitInfo(num_data_in_leaf, sum_gradients, sum_hessians);
            isFinish = booster->ContinueTrain();
            if(isFinish){
                ConstantSP dumpModel = new String(booster->getModel());
                mapRet->set(0, new Char(5));// task = 5 dumpModel
                mapRet->set(1, dumpModel);
                mapResult->set(1, new Void());
                delete booster;
            }
            else{
                status = booster->getTrainStatus();
                if(status == LightGBM::TrainStatus::sumHistograms){
                    vector<LightGBM::FeatureHistogram *> featuresHistograms = booster->getHisgotram();
                    ConstantSP histogramsVec = Util::createVector(DT_ANY, featuresHistograms.size());
                    for(size_t i = 0; i < featuresHistograms.size(); i++){
                        if(featuresHistograms[i] == nullptr){
                           histogramsVec->set(i, new Void());
                           continue;
                        }
                        int size = featuresHistograms[i]->SizeOfHistgram() / sizeof(double);
                        ConstantSP vec = Util::createVector(DT_DOUBLE, size);
                        vec->setDouble(0, size,featuresHistograms[i]->RawData());
                        histogramsVec->set(i, vec);
                    }
                    mapRet->set(0, new Char(2));// task = 2 gather Histograms
                    mapRet->set(1, histogramsVec);
                }
                else if(status == LightGBM::TrainStatus::NoneTrain){
                    isFinish = booster->TrainOneIter();
                    if(isFinish){
                        ConstantSP dumpModel = new String(booster->getModel());
                        mapRet->set(0, new Char(5));// task = 5 dumpModel
                        mapRet->set(1, dumpModel);
                        mapResult->set(1, new Void());
                        delete booster;
                    }
                    else{
                        status = booster->getTrainStatus();
                    
                        if(status == LightGBM::TrainStatus::sumSplitInfo){
                            ConstantSP res = Util::createVector(DT_ANY, 3);
                            int num_data_in_leaf;
                            double sum_gradients;
                            double sum_hessians;
                            booster->getSplitInfo(&num_data_in_leaf, &sum_gradients, &sum_hessians);
                            res->set(0, new Int(num_data_in_leaf));
                            res->set(1, new Double(sum_gradients));
                            res->set(2, new Double(sum_hessians));
                            mapRet->set(0, new Char(1)); // task = 1 sum SplitInfo
                            mapRet->set(1, res);
                        }
                        else{
                            throw RuntimeException("Error train status");
                        }
                    }
                    
                }
            }
        }
        else if(status == LightGBM::TrainStatus::sumHistograms){
            vector<LightGBM::FeatureHistogram*> res = booster->getHisgotram();
            ConstantSP FeatureConst = initValue->get(1);
            for(size_t i = 0; i < res.size(); i++){
                ConstantSP histogram = FeatureConst->get(i);
                if(histogram->isNull()) continue;
                double* histogramData = (double *)histogram->getDataArray();
                res[i]->FromMemory((char*)histogramData);
            }
            isFinish = booster->ContinueTrain();
            if(isFinish){
                ConstantSP dumpModel = new String(booster->getModel());
                mapRet->set(0, new Char(5));// task = 5 dumpModel
                mapRet->set(1, dumpModel);
                mapResult->set(1, new Void());
                delete booster;
            }
            else{
                status = booster->getTrainStatus();
                if(status == LightGBM::TrainStatus::sumHistograms){
                    vector<LightGBM::FeatureHistogram *> featuresHistograms = booster->getHisgotram();
                    ConstantSP histogramsVec = Util::createVector(DT_ANY, featuresHistograms.size());
                     for(size_t i = 0; i < featuresHistograms.size(); i++){
                        if(featuresHistograms[i] == nullptr){
                           histogramsVec->set(i, new Void());
                           continue;
                        }
                        int size = featuresHistograms[i]->SizeOfHistgram() / sizeof(double);
                        ConstantSP vec = Util::createVector(DT_DOUBLE, size);
                        vec->setDouble(0, size,featuresHistograms[i]->RawData());
                        histogramsVec->set(i, vec);
                    }
                    mapRet->set(0, new Char(2));// task = 2 gather Histograms
                    mapRet->set(1, histogramsVec);
                }
                else if(status == LightGBM::TrainStatus::NoneTrain){
                    isFinish = booster->TrainOneIter();
                    if(isFinish){
                        ConstantSP dumpModel = new String(booster->getModel());
                        mapRet->set(0, new Char(5));// task = 5 dumpModel
                        mapRet->set(1, dumpModel);
                        mapResult->set(1, new Void());
                        delete booster;
                    }
                    else{
                        status = booster->getTrainStatus();
                    
                        if(status == LightGBM::TrainStatus::sumSplitInfo){
                            ConstantSP res = Util::createVector(DT_ANY, 3);
                            int num_data_in_leaf;
                            double sum_gradients;
                            double sum_hessians;
                            booster->getSplitInfo(&num_data_in_leaf, &sum_gradients, &sum_hessians);
                            res->set(0, new Int(num_data_in_leaf));
                            res->set(1, new Double(sum_gradients));
                            res->set(2, new Double(sum_hessians));
                            mapRet->set(0, new Char(1)); // task = 1 sum SplitInfo
                            mapRet->set(1, res);
                        }
                        else{
                            throw RuntimeException("Error train status");
                        }
                    }
                }
                else{
                    throw RuntimeException("Error train status");
                }
            }
        }
    }
    return mapResult;

}

ConstantSP lgbmReduce(const ConstantSP& res1, const ConstantSP& res2){
    char task = res1->get(0)->getChar();
    switch (task)
    {
    case 5:// task = 5 dump model
        return res1;
    case 0:{// task = 0 gather binMappers

        if(res1->get(1)->get(0)->getType() == DT_ANY){
            ((VectorSP)res1->get(1))->append(res2->get(1));

        }
        else{
            VectorSP res;
            res = Util::createVector(DT_ANY, 2);
            res->set(0, res1->get(1));
            res->set(1, res2->get(1));
            res1->set(1, res);
        }
        return res1;
    }
    case 1:// task = 2 sum SplitInfo
        res1->get(1)->set(0, new Int(res1->get(1)->get(0)->getInt() + res2->get(1)->get(0)->getInt()));
        res1->get(1)->set(1, new Double(res1->get(1)->get(1)->getDouble() + res2->get(1)->get(1)->getDouble()));
        res1->get(1)->set(2, new Double(res1->get(1)->get(2)->getDouble() + res2->get(1)->get(2)->getDouble()));
        return res1;
    case 2:// task = 2 gather Histograms
        res1->set(1, OperatorImp::add(res1->get(1), res2->get(1)));
        return res1;
    // case 3:{
    //     ConstantSP small1 = res1->get(1)->get(0);
    //     ConstantSP large1 = res1->get(1)->get(1);
    //     ConstantSP small2 = res2->get(1)->get(0);
    //     ConstantSP large2 = res2->get(1)->get(1);
    //     LightGBM::LightSplitInfo p1, p2;
    //     p1.CopyFrom((char*)small1->getDataArray());
    //     p2.CopyFrom((char*)small2->getDataArray());
    //     if (p2 > p1) {
    //         res1->get(1)->set(0, small2);
    //     }
    //     p1.CopyFrom((char*)large1->getDataArray());
    //     p2.CopyFrom((char*)large2->getDataArray());
    //     if (p2 > p1) {
    //         res1->get(1)->set(1, large2);
    //     }
    //     return res1;
    // }
    default:
        throw RuntimeException("wrong task");
    }

}

ConstantSP lgbmTerminate(const ConstantSP& res1, const ConstantSP& res2){
    if(res2->get(0)->getChar() == 5){
        return new Bool(true);
    }
    return new Bool(false);

}

vector<int> getSampledIndex(LightGBM::Config& config, const std::vector<std::vector<double>>& data){
    LightGBM::Random random(config.data_random_seed);
    int sample_cnt = config.bin_construct_sample_cnt;
    if (static_cast<size_t>(sample_cnt) > data.size()) {
        sample_cnt = static_cast<int>(data.size());
    }
    auto sample_indices = random.Sample(static_cast<int>(data.size()), sample_cnt);
    
    return sample_indices;
}

Dataset* constructDatasetWithBinMapper(vector<string>& header, ConstantSP binMappers, LightGBM::Config& config, const std::vector<std::vector<double>>& data, vector<int>& sampledIndex){
    vector<string> feature_names_ = header;
    std::string name_prefix("name:");
    int label_idx_;

    if (config.label_column.size() > 0) {
      if (Common::StartsWith(config.label_column, name_prefix)) {
        std::string name = config.label_column.substr(name_prefix.size());
        label_idx_ = -1;
        for (int i = 0; i < static_cast<int>(feature_names_.size()); ++i) {
          if (name == feature_names_[i]) {
            label_idx_ = i;
            break;
          }
        }
        if (label_idx_ >= 0) {
          Log::Info("Using column %s as label", name.c_str());
        } else {
          Log::Fatal("Could not find label column %s in data file \n"
                     "or data file doesn't contain header", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config.label_column.c_str(), &label_idx_)) {
          Log::Fatal("label_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as label", label_idx_);
      }
    }

    if (!feature_names_.empty()) {
      // erase label column name
      feature_names_.erase(feature_names_.begin() + label_idx_);
    }

    auto dataset = std::unique_ptr<LightGBM::Dataset>(new LightGBM::Dataset());
    dataset->label_idx_ = label_idx_;
   
    dataset->num_data_ = static_cast<data_size_t>(data.size());
   // ConstantSP indices = Util::createVector(DT_INDEX, sampledIndex.size(), sampledIndex.size(), true);
    std::vector<std::vector<double>> sample_data(sampledIndex.size());
    for (size_t i = 0; i < sampledIndex.size(); ++i) {
        const size_t idx = sampledIndex[i];
        sample_data[i] = data[idx];
    }
    std::vector<std::vector<double>> sample_values;
    std::vector<std::vector<int>> sample_indices;
    std::vector<std::pair<int, double>> oneline_features;
    int col=sample_data[0].size();
    std::vector<LightGBM::data_size_t> used_data_indices;
    for (int i = 0; i < static_cast<int>(sample_data.size()); ++i) {
            oneline_features.clear();
        // parse features
    // parser->ParseOneLine(sample_data[i].c_str(), &oneline_features, &label);

        for(int idx=0;idx<col-1;idx++)
            oneline_features.push_back(std::make_pair(idx,sample_data[i][idx]));
        for (std::pair<int, double>& inner_data : oneline_features) {
            if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
                sample_values.resize(inner_data.first + 1);
                sample_indices.resize(inner_data.first + 1);
            }
            if (std::fabs(inner_data.second) > LightGBM::kZeroThreshold || std::isnan(inner_data.second)) {
                sample_values[inner_data.first].emplace_back(inner_data.second);
                sample_indices[inner_data.first].emplace_back(i);
            }
        }
    }
    int num_total_features = sample_values.size();
    dataset->feature_groups_.clear();
    dataset->num_total_features_ = static_cast<int>(sample_values.size());
    dataset->set_feature_names(feature_names_);
    std::vector<std::vector<double>> forced_bin_bounds(num_total_features, std::vector<double>());
    std::vector<std::unique_ptr<LightGBM::BinMapper>> bin_mappers(num_total_features);
    //const LightGBM::data_size_t filter_cnt = static_cast<LightGBM::data_size_t>(static_cast<double>(config.min_data_in_leaf* sample_data.size()) / data.size());

    for (int i = 0; i < static_cast<int>(binMappers->size()); ++i){
        ConstantSP map = binMappers->get(i);
        bin_mappers[i].reset(new LightGBM::BinMapper());
        bin_mappers[i]->CopyFrom((char*)map->getDataArray());
    }

    dataset->Construct(&bin_mappers, dataset->num_total_features_, forced_bin_bounds, Common::Vector2Ptr<int>(&sample_indices).data(),
                     Common::Vector2Ptr<double>(&sample_values).data(),
                     Common::VectorSize<int>(sample_indices).data(), static_cast<int>(sample_indices.size()), sample_data.size(), config);


    dataset->metadata_.Init(dataset->num_data_, -1, -1);
          // extract features


    //OMP_INIT_EX();
    // if doesn't need to prediction with initial model
    //#pragma omp parallel for schedule(static) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < dataset->num_data_; ++i) {
      //OMP_LOOP_EX_BEGIN();
      oneline_features.clear();
      // parser
     // parser->ParseOneLine(ref_text_data[i].c_str(), &oneline_features, &tmp_label);
     for(int idx=0;idx<col-1;idx++)
          oneline_features.push_back(std::make_pair(idx,data[i][idx]));
      // set label
      dataset->metadata_.SetLabelAt(i, static_cast<label_t>(data[i][col-1]));
      // free processed line:
      //ref_text_data[i].clear();
      // shrink_to_fit will be very slow in linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      std::vector<bool> is_feature_added(dataset->num_features_, false);
      // push data
      for (auto& inner_data : oneline_features) {
        if (inner_data.first >= dataset->num_total_features_) { continue; }
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          is_feature_added[feature_idx] = true;
          // if is used feature
          int group = dataset->feature2group_[feature_idx];
          int sub_feature = dataset->feature2subfeature_[feature_idx];
          dataset->feature_groups_[group]->PushData(0, sub_feature, i, inner_data.second);
        } else {
          if (inner_data.first == -1) {
            dataset->metadata_.SetWeightAt(i, static_cast<label_t>(inner_data.second));
          } else if (inner_data.first == -1) {
            dataset->metadata_.SetQueryAt(i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
      dataset->FinishOneRow(0, i, is_feature_added);
      //OMP_LOOP_EX_END();
    }
    //OMP_THROW_EX();
  
    dataset->FinishLoad();
    dataset->metadata_.CheckOrPartition(data.size(), used_data_indices);
    // need to check training data
    return dataset.release();


}

ConstantSP constructBinMapper(LightGBM::Config& config, vector<string>& header,  const std::vector<std::vector<double>>& data, vector<int>& sampledIndex){
    std::map<std::string, int> name2idx;
      for (size_t i = 0; i < header.size(); ++i) {
        name2idx[header[i]] = static_cast<int>(i);
      }
    

    std::set<int> categorical_features_;
    std::string name_prefix("name:");
    if (config.categorical_feature.size() > 0) {
        if (Common::StartsWith(config.categorical_feature, name_prefix)) {
            std::string names = config.categorical_feature.substr(name_prefix.size());
            for (auto name : Common::Split(names.c_str(), ',')) {
                if (name2idx.count(name) > 0) {
                int tmp = name2idx[name];
                categorical_features_.emplace(tmp);
                } else {
                Log::Fatal("Could not find categorical_feature %s in data file", name.c_str());
                }
            }
        } else {
            for (auto token : Common::Split(config.categorical_feature.c_str(), ',')) {
                int tmp = 0;
                if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
                Log::Fatal("categorical_feature is not a number,\n"
                            "if you want to use a column name,\n"
                            "please add the prefix \"name:\" to the column name");
                }
                categorical_features_.emplace(tmp);
            }
        }
    }

    std::vector<std::vector<double>> sample_data(sampledIndex.size());
    for (size_t i = 0; i < sampledIndex.size(); ++i) {
        const size_t idx = sampledIndex[i];
        sample_data[i] = data[idx];
    }
    std::vector<std::vector<double>> sample_values;
    std::vector<std::vector<int>> sample_indices;
    std::vector<std::pair<int, double>> oneline_features;
    int col=sample_data[0].size();
    for (int i = 0; i < static_cast<int>(sample_data.size()); ++i) {
            oneline_features.clear();
        // parse features
    // parser->ParseOneLine(sample_data[i].c_str(), &oneline_features, &label);

        for(int idx=0;idx<col-1;idx++){
            oneline_features.push_back(std::make_pair(idx,sample_data[i][idx]));
        }
        for (std::pair<int, double>& inner_data : oneline_features) {
            if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
                sample_values.resize(inner_data.first + 1);
                sample_indices.resize(inner_data.first + 1);
            }
            if (std::fabs(inner_data.second) > LightGBM::kZeroThreshold || std::isnan(inner_data.second)) {
                sample_values[inner_data.first].emplace_back(inner_data.second);
                sample_indices[inner_data.first].emplace_back(i);
            }
        }
    }
    int num_total_features = sample_values.size();
    std::vector<std::vector<double>> forced_bin_bounds(num_total_features, std::vector<double>());
    std::vector<std::unique_ptr<LightGBM::BinMapper>> bin_mappers(num_total_features);
    const LightGBM::data_size_t filter_cnt = static_cast<LightGBM::data_size_t>(
        static_cast<double>(config.min_data_in_leaf* sample_data.size()) / data.size());
    // start find bins
    // if only one machine, find bin locally
    //OMP_INIT_EX();
    //#pragma omp parallel for schedule(guided)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      //OMP_LOOP_EX_BEGIN();
      LightGBM::BinType bin_type = LightGBM::BinType::NumericalBin;
      if (categorical_features_.count(i)) {
        bin_type = LightGBM::BinType::CategoricalBin;
      }
      bin_mappers[i].reset(new LightGBM::BinMapper());
      if (config.max_bin_by_feature.empty()) {
        bin_mappers[i]->FindBin(sample_values[i].data(), static_cast<int>(sample_values[i].size()),
                                sample_data.size(), config.max_bin, config.min_data_in_bin,
                                filter_cnt, config.feature_pre_filter, bin_type, config.use_missing, config.zero_as_missing,
                                forced_bin_bounds[i]);
      } else {
        bin_mappers[i]->FindBin(sample_values[i].data(), static_cast<int>(sample_values[i].size()),
                                sample_data.size(), config.max_bin_by_feature[i],
                                config.min_data_in_bin, filter_cnt, config.feature_pre_filter, bin_type, config.use_missing,
                                config.zero_as_missing, forced_bin_bounds[i]);
      }
      //OMP_LOOP_EX_END();
    }
    //OMP_THROW_EX();
    ConstantSP binMappers = Util::createVector(DT_ANY, sample_values.size());
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i){
        ConstantSP map = Util::createVector(DT_CHAR, bin_mappers[i]->SizesInByte());
        bin_mappers[i]->CopyTo((char*)map->getDataArray());
        binMappers->set(i, map);
    }
    return binMappers;

}

ConstantSP lgbmFinal(Heap *heap, vector<ConstantSP> &args){
    ConstantSP res = args[1];
    if(res->get(0)->getChar() == 0){// task = 0 gather binMapper
        ConstantSP binMappers = res->get(1);
        int machines = binMappers->size();
        int feature = binMappers->get(0)->size();
        int group = (feature + machines - 1) / machines;
        ConstantSP binMappersGlobal = Util::createVector(DT_ANY, feature);
        int sumGroup = 0;
        for(int i = 0; i < machines; i++){
            if(sumGroup >= feature) break;
            for(int j = 0; j < group; j++){
                if(sumGroup >= feature) break;
                binMappersGlobal->set(sumGroup, binMappers->get(i)->get(sumGroup));
                sumGroup ++;
            }
        }
        res->set(1, binMappersGlobal);
       
    }
    return res;
}