/*
 * statsmodels.h
 *
 * Created on: Jan 8.2021
 *     Author: zkluo
 *
 */
#ifndef STATSMODELS_LIBRARY_H
#define STATSMODELS_LIBRARY_H
#include <CoreConcept.h>

extern "C" ConstantSP adfuller(Heap* heap,vector<ConstantSP>& args);
extern "C" ConstantSP kpss(Heap* heap,vector<ConstantSP>& args);
extern "C" ConstantSP bds(Heap* heap,vector<ConstantSP>& args);
extern "C" ConstantSP q_stat(Heap* heap,vector<ConstantSP>& args);
#endif //STATSMODELS_LIBRARY_H
