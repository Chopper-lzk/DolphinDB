/*
 * lgbm.h
 *
 * Created on: Dec 8.2020
 *     Author: zkluo
 *
 */

#ifndef _LGBM_H_
#define _LGBM_H_
#include "CoreConcept.h"
#include <vector>
#include <string>
extern "C"  ConstantSP modelTrain(Heap* heap, vector<ConstantSP>& args);
extern "C"  ConstantSP modelPredict(Heap* heap, vector<ConstantSP>& args);
extern "C"  ConstantSP modelSave(Heap* heap, vector<ConstantSP>& args);
extern "C"  ConstantSP modelLoad(Heap* heap, vector<ConstantSP>& args);
extern "C"  ConstantSP modelTrainKFold(Heap* heap, vector<ConstantSP>& args);
extern "C"  ConstantSP fitGoodness(Heap* heap,vector<ConstantSP>& args);
extern "C"  ConstantSP modelTrainDS(Heap* heap, vector<ConstantSP>& args);
ConstantSP lgbmMap(Heap* heap, vector<ConstantSP>& args);
ConstantSP lgbmReduce(const ConstantSP& res1, const ConstantSP& res2);
ConstantSP lgbmTerminate(const ConstantSP& res1, const ConstantSP& res2);
ConstantSP lgbmFinal(Heap *heap, vector<ConstantSP> &args);

#endif //_LGBM_H_
