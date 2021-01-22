# DolphinDB
g++ -DLINUX -fPIC -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -c src/Demo.cpp -I../include -o Demo.o
g++ -fPIC -shared -o libPluginTest.so Test.o -lDolphinDB -L/home/zkluo/DolphinDB_Linux64_V1.20.8/server 

