/*
 * statsmodels.cpp
 *
 * Created on: Jan 8.2021
 *     Author: zkluo
 *
 */

#include "statsmodels.h"
#include <math.h>
#include <Util.h>
#include <ScalarImp.h>
#include <omp.h>
#include <stdio.h>

static void lagMat(vector<double>& ,int,vector<vector<double>>& );
static void addTrend(vector<vector<double>>&,string);
static double mackinnonp(Heap*,double ,string);
static TableSP olsCall(Heap* ,vector<double>& ,vector<vector<double>>& );
static void residGet(TableSP ,vector<double>&, vector<vector<double>>& ,vector<double>&);
static ConstantSP cdfCall(Heap* ,double );
static vector<double> mackinnoncrit(string ,int);
static double kpssAutoLag(vector<double>&);
static double sigmaEstKpss(vector<double>&,int);
static double interp(double,vector<double>&,vector<double>&);

ConstantSP adfuller(Heap* heap,vector<ConstantSP>& args){
    string funcName="adfuller";
    string syntax="Usage: "+funcName+"(x,[maxLag],[regression=\"c\"])";
    if(args.size()<1||args.size()>3)
        throw IllegalArgumentException(funcName,syntax+" Illegal arguments size.");
    if(!(args[0]->isVector() && args[0]->isNumber()))
        throw IllegalArgumentException(funcName,syntax+" x must be a numeric vector.");
    vector<double> x(args[0]->size(),0);
    args[0]->getDouble(0,args[0]->size(),&x[0]);
    int maxLag;
    string regression;
    if(args.size()>=3) {
        if(!(args[2]->getType() == DT_STRING && args[2]->isScalar()))
            throw IllegalArgumentException(funcName,syntax+" regression should be a string.");
        regression = args[2]->getString();
        if(!(regression=="c"||regression=="ct"||regression=="ctt"||regression==""))
            throw IllegalArgumentException(funcName,syntax+" Illegal regression value, should be {\"c\",\"ct\",\"ctt\",\"\"}.");
    } else regression="c";
    int nobs=x.size();
    int ntrend=regression.size();
    // maxlag process
    if(args.size()<2){
        maxLag=static_cast<int>(ceil(12.0*pow(nobs/100.0,1/4.0)));
        maxLag=nobs/2-ntrend-1<maxLag?nobs/2-ntrend-1:maxLag;
        if(maxLag<0)
            throw IllegalArgumentException(funcName,syntax+" sample size is too short to use selected regression component");
    }
    else{
        if(!(args[1]->isScalar() && args[1]->getCategory() == INTEGRAL))
            throw IllegalArgumentException(funcName,syntax+" maxLag should be an integer.");
        maxLag=args[1]->getInt();
        if(maxLag<0)
            throw IllegalArgumentException(funcName,syntax+" sample size is too short to use selected regression component");
        if(maxLag>(nobs/2-ntrend-1))
            throw IllegalArgumentException(funcName,syntax+" maxlag must be less than (nobs/2 - 1 - ntrend) where n trend is the number of included deterministic regressors");
    }

    //--------------------end of argument validation----------------------------

    vector<double> xdiff(x.size()-1,0);
    vector<vector<double>> xdall;
    vector<double> xdshort;
    for(int i=0;i<xdiff.size();i++) xdiff[i]=x[i+1]-x[i];
    lagMat(xdiff,maxLag,xdall);
    nobs=xdall.size();
    for(int idx=0;idx<nobs;idx++) xdall[idx][0]=x[x.size()-nobs-1+idx];
    for(int idx=xdiff.size()-nobs;idx<xdiff.size();idx++) xdshort.push_back(xdiff[idx]);

    int usedLag=maxLag;

    addTrend(xdall,regression);
    TableSP resols=olsCall(heap,xdshort,xdall);
    double adfstat=resols->getColumn("tstat")->getDouble(0);
    double pValue=mackinnonp(heap,adfstat,regression);
    vector<double> res=mackinnoncrit(regression,nobs);
    vector<string> colNames={"adfStat","pValue","usedLag","nobs","1%","5%","10%"};
    vector<ConstantSP> colValues={new Double(adfstat),new Double(pValue),new Int(usedLag),new Int(nobs),new Double(res[0]),new Double(res[1]),new Double(res[2])};
    TableSP table=Util::createTable(colNames,colValues);
    return table;
}
static void lagMat(vector<double>& x,int maxLag,vector<vector<double>>& res){
    int nobs=x.size();
    if(maxLag>=nobs)
        throw RuntimeException("maxLag should smaller than nobs.");
    vector<vector<double>> lm(nobs+maxLag,vector<double>(maxLag+1,0));
#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for (int k = 0; k <= maxLag; k++) {
        for (int i = 0; i < nobs; i++)
            lm[maxLag - k + i][maxLag - k] = x[i];
    }
    for(int i=maxLag;i<nobs;i++)
        res.push_back(lm[i]);
}
static void addTrend(vector<vector<double>>& x,string trend){
#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for(int idx=0;idx<x.size();idx++){
        double base=1;
        for(int k=0;k<trend.size();k++) {
            x[idx].push_back(base);
            base*=(idx+1);
        }
    }
}
static TableSP olsCall(Heap* heap,vector<double>& y,vector<vector<double>>& x){
    FunctionDefSP ols = heap->currentSession()->getFunctionDef("ols");
    ConstantSP Y=Util::createVector(DT_DOUBLE,y.size());
    Y->setDouble(0,y.size(),&y[0]);
    ConstantSP X=Util::createDoubleMatrix(x[0].size(),x.size());
    ConstantSP _col=Util::createVector(DT_DOUBLE,x.size());
    for(int i=0;i<x[0].size();i++){
        for(int j=0;j<x.size();j++)
            _col->setDouble(j,x[j][i]);
        X->setColumn(i,_col);
    }
    vector<ConstantSP> arguments = {Y, X,new Int(0),new Int(1)};
    TableSP res = ols->call(heap, arguments);
    return res;
}
static void residGet(TableSP t,vector<double>&y, vector<vector<double>>& x,vector<double>& resid){
    vector<double> beta(x[0].size(),0);
    t->getColumn("beta")->getDouble(0,beta.size(),&beta[0]);
    for(int idx=0;idx<x.size();idx++){
        resid[idx]=0;
        for(int k=0;k<beta.size();k++)
            resid[idx]+=beta[k]*x[idx][k];
        resid[idx]=y[idx]-resid[idx];
    }
}
static double mackinnonp(Heap* heap,double teststat,string regression){
    vector<double> _tau_maxs={INFINITY,2.47,0.7,0.54};
    vector<double> _tau_mins={-19.04,-18.83,-16.18,-17.17};
    vector<double> _tau_stars={-1.04,-1.61,-2.89,-3.21};
    double scalingL=1e-1;
    double scalingS=1e-2;
    vector<vector<double>> _tau_smallps={{0.6344,1.2378,3.2496*scalingS},{2.1659,1.4412,3.8296*scalingS},{3.2512,1.6047,4.9588*scalingS},{4.0003,1.648,4.8288*scalingS}};
    vector<vector<double>> _tau_largeps={{0.4797, 9.3557*scalingL, -0.6999*scalingL, 3.3066*scalingS},{1.7339, 9.3202*scalingL, -1.2745*scalingL, -1.0368*scalingS},
                                        {2.5261, 6.1654*scalingL, -3.7956*scalingL, -6.0285*scalingS},{3.0778, 4.9529*scalingL, -4.1477*scalingL, -5.9359*scalingS}};
    if(teststat>_tau_maxs[regression.size()])
        return 1.0;
    else if(teststat<_tau_mins[regression.size()])
        return 0.0;
    vector<double> tau_coef;
    if(teststat<=_tau_stars[regression.size()])
        tau_coef=_tau_smallps[regression.size()];
    else
        tau_coef=_tau_largeps[regression.size()];
    double res=0;
    for(int i=tau_coef.size()-1;i>=0;i--)
        res=res*teststat+tau_coef[i];
    res=cdfCall(heap,res)->getDouble();
    return res;
}
static ConstantSP cdfCall(Heap* heap,double x){
    FunctionDefSP cdfNorm=heap->currentSession()->getFunctionDef("cdfNormal");
    vector<ConstantSP> arguments={new Int(0),new Int(1),new Double(x)};
    return cdfNorm->call(heap,arguments);
}
static vector<double> mackinnoncrit(string regression,int nobs){
    vector<vector<vector<double>>> tau_2010s = {{{-2.56574, -2.2358, -3.627, 0},{-1.94100, -0.2686, -3.365, 31.223},{-1.61682, 0.2656, -2.714, 25.364}},
                                                {{-3.43035, -6.5393, -16.786, -79.433},{-2.86154, -2.8903, -4.234, -40.040},{-2.56677, -1.5384, -2.809, 0}},
                                                {{-3.95877, -9.0531, -28.428, -134.155},{-3.41049, -4.3904, -9.036, -45.374},{-3.12705, -2.5856, -3.925, -22.380}},
                                                {{-4.37113, -11.5882, -35.819, -334.047},{-3.83239, -5.9057, -12.490, -118.284},{-3.55326, -3.6596, -5.293, -63.559}}};
    double base=1.0/nobs;
    int index=regression.size();
    vector<double> res(3,0);
    vector<string> colNames={"1%","5%","10%"};
    for(int i=0;i<tau_2010s[index].size();i++){
        for(int j=tau_2010s[index][i].size()-1;j>=0;j--)
            res[i]=res[i]*base+tau_2010s[index][i][j];
    }
    return res;
}
ConstantSP kpss(Heap* heap,vector<ConstantSP>& args){
    string funcName="kpss";
    string syntax="Usage: "+funcName+"(x,[regression=\"c\"],[nLags=\"legacy\"])";
    if(args.size()<1||args.size()>3)
        throw IllegalArgumentException(funcName,syntax+" Illegal argument size.");
    if(!(args[0]->isVector() && args[0]->isNumber()&& args[0]->size()>0))
        throw IllegalArgumentException(funcName,syntax+" x must be a nonempty numeric vector.");
    vector<double> x(args[0]->size(),0);
    args[0]->getDouble(0,args[0]->size(),&x[0]);
    int nobs=x.size();
    string regression;
    int nLags;
    if(args.size()<2)
        regression="c";
    else{
        if(args[1]->getType() == DT_STRING && args[1]->isScalar())
            regression=args[1]->getString();
        else
            throw IllegalArgumentException(funcName,syntax+" regression should be a string.");
    }
    if(regression!="c"&&regression!="ct")
        throw IllegalArgumentException(funcName,syntax+" Illegal regression value, should be \"c\" or \"ct\".");
    vector<double> resids(nobs,0);
    vector<double> crit;
    if(regression=="ct"){
        vector<vector<double>> constantVec(nobs,vector<double>(2,1));
        for(int i=0;i<nobs;i++) constantVec[i][1]=i+1;
        TableSP t=olsCall(heap,x,constantVec);
        residGet(t,x,constantVec,resids);
        vector<double> _item={0.119,0.146,0.176,0.216};
        crit=_item;
    }
    else{
        double mean=0;
        for(int idx=0;idx<nobs;idx++) mean+=x[idx]/nobs;
        for(int idx=0;idx<nobs;idx++) resids[idx]=x[idx]-mean;
        vector<double> _item={0.347,0.463,0.574,0.739};
        crit=_item;
    }
    if(args.size()<3){
        nLags=static_cast<int>(ceil(12.0*pow(nobs/100.0,1/4.0)));
        nLags=nLags<nobs-1?nLags:nobs-1;
    }
    else{
        if(args[2]->getType()==DT_STRING&&args[2]->isScalar()){
            if(args[2]->getString()=="legacy"){
                nLags=static_cast<int>(ceil(12.0*pow(nobs/100.0,1/4.0)));
                nLags=nLags<nobs-1?nLags:nobs-1;
            }
            else if(args[2]->getString()=="auto"){
                nLags=kpssAutoLag(resids);
                nLags=nLags<nobs-1?nLags:nobs-1;
            }
            else
                throw IllegalArgumentException(funcName,syntax+" Illegal nLags string, should be \"legacy\" or \"auto\".");
        }
        else if(args[2]->getType()==DT_INT&&args[2]->isScalar())
            nLags=args[2]->getInt();
        else
            throw IllegalArgumentException(funcName,syntax+" Illegal nLags value, should be a string or interger.");
    }
    if(nLags>=nobs)
        throw IllegalArgumentException(funcName,syntax+" too large nLags value, should be smaller than nobs.");
    //-------------------end of arguments validation--------------------------------

    vector<double> pvals={0.10,0.05,0.025,0.01};
    double eta=0;
    double cum=0;
    for(int idx=0;idx<nobs;idx++){
        cum+=resids[idx];
        eta+=cum*cum/(nobs*nobs);
    }
    double sHat=sigmaEstKpss(resids,nLags);
    double kpssStat=eta/sHat;
    double pValue=interp(kpssStat,crit,pvals);
    char warnMsg[]="The test statistic is outside of the range of p-values available in the look-up table. The actual p-value is %s than the p-value returned.";
    if(pValue==pvals[pvals.size()-1])
        printf(warnMsg,"smaller");
    if(pValue==pvals[0])
        printf(warnMsg,"greater");
    vector<string> colNames={"kpssStat","pValue","nLags","10%","5%","2.5%","1%"};
    vector<ConstantSP> values={new Double(kpssStat),new Double(pValue),new Int(nLags),new Double(crit[0]),new Double(crit[1]),new Double(crit[2]),new Double(crit[3])};
    TableSP res=Util::createTable(colNames,values);
    return res;
}
static double kpssAutoLag(vector<double>& resids){
    int nobs=resids.size();
    int covLags=static_cast<int>(pow(nobs,2.0/9.0));
    double s0,s1=0;
    for(int i=0;i<nobs;i++) s0+=resids[i]*resids[i]/nobs;
    for(int i=1;i<=covLags;i++){
        double residsProd=0;
        for(int k=0;i+k<nobs;k++) residsProd+=resids[i+k]*resids[k];
        residsProd/=(nobs/2.0);
        s0+=residsProd;
        s1+=i*residsProd;
    }
    double sHat=s1/s0;
    double pwr=1.0/3.0;
    double gamma_hat=1.1447*pow(sHat*sHat,pwr);
    int autoLags=static_cast<int>(gamma_hat*pow(nobs,pwr));
    return autoLags;
}
static double sigmaEstKpss(vector<double>& resids,int nLags){
    int nobs=resids.size();
    double sHat=0;
    for(int i=0;i<resids.size();i++) sHat+=resids[i]*resids[i];
    for(int k=1;k<=nLags;k++){
        double residsProd=0;
        for(int j=0;j+k<nobs;j++) residsProd+=resids[k+j]*resids[j];
        sHat+=2*residsProd*(1.0-(k/(nLags+1.0)));
    }
    return sHat/nobs;
}
static double interp(double x,vector<double>& xp,vector<double>& fp){
    int idx=0;
    for(;idx<xp.size()&&x>xp[idx];idx++){}
    if(idx==0)
        return fp[0];
    else if(idx==xp.size())
        return fp[idx-1];
    else{
        return fp[idx-1]+(fp[idx]-fp[idx-1])*(x-xp[idx-1])/((xp[idx]-xp[idx-1]));
    }
}