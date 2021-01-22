# 服务器免密登陆
1. 客户端生成公私钥
```
ssh-keygen
cd ~/.ssh
ls
```
2. 上传公钥到服务器
```
ssh-copy-id -i ~/.ssh/id_rsa.pub [user]@[ip]
cd ~/.ssh
```
3. 文件传输
- 本地到服务器
scp /path/filename username@servername:/path/
- 服务器到本地
scp username@servername:/path/filename /path/

4. `GLIBCXX_x.x.x' not found问题
- 覆盖本地libstdc++到指定目录path
- 配置环境变量: export LD_LIBRARY_PATH="path"

# DolphinDB 分布式数据库创建
```
n=1000000;
t=table(rand(`IBM`MS`APPL`AMZN,n) as symbol, rand(10.0, n) as value)
db = database("dfs://rangedb_tradedata", RANGE, `A`F`M`S`ZZZZ)
Trades = db.createPartitionedTable(t, "Trades", "symbol");
Trades.append!(t);
```
