# Fusion Mind Rag

## 文件说明

### src/agent

里面的 seeker，inspector，synthesizer 是三个主要 agent，其调用和运行都封装在 orchestrator.py 里.
orchestrator 是整个 RAP Pipeline 的整体封装.
map_dict, agent_prompt 是构建 agents prompt 所需要的组件.

### src/llms

都是其他模块需要用到的 llm 组件，可以不用看.
其中 llm.py 封装了需要用到 vlm 模型. 通过 QwenAPI 来调用. 
vl_embedding 是 search engine 所需要的嵌入模型.

### src/models

gumbel 网络的实现.

### src/searcher

SearchEngine 搜索引擎.
分为三个不同的 Text，Image，Table Searcher. 都继承自 BaseSearcher.
SearchEngine 负责根据 Gumbel 决策的模态来调用其一 Searcher 进行检索.

### src/training

gumbel 网络的训练. 可以不用运行. 直接使用已经训练好的 checkpoints. 

### src utils

一些工具函数，主要负责格式转换和 query 的嵌入.

## 运行

先根据 environment.yml 来下载所需要的依赖.
直接在文件夹下运行 python run.py 即可.



