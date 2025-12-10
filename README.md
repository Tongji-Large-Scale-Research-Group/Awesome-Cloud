# Awesome Cloud

<!-- 中文 | [English](README_en.md) -->

一个关于云计算领域中最优质的论文、工具和信息的精选列表，适合开发者和研究人员参考。

## 周刊

* 研究团队会定期进行科研进展的交流，并分享领域前沿知识（见[待讨论话题清单](#待讨论话题清单)）。具体细节请参见[团队交流指南](./docs/communication.md)。
* 每周讨论内容会形成概要周刊，便于回顾与内容沉淀。具体细节参见[**周刊**](./docs/weekly.md)。
* 不定期开展复盘会议，为了更好地跟踪会议中提出的问题和改进建议，我们设立了 **[问题追踪记录](./docs/tracking.md)** 文件，请大家及时关注并更新。


## 目录
- [Awesome Cloud](#awesome-cloud)
  - [周刊](#周刊)
  - [目录](#目录)
  - [已讨论话题索引](#已讨论话题索引)
    - [学术界](#学术界)
    - [工业界 or 实践](#工业界-or-实践)
  - [待讨论话题清单](#待讨论话题清单)
    - [一、学术界](#一学术界)
      - [1.1 前沿论文](#11-前沿论文)
      - [1.2 顶级期刊](#12-顶级期刊)
      - [1.3 顶级会议](#13-顶级会议)
      - [1.4 高校团队](#14-高校团队)
    - [二、工业界](#二工业界)
      - [2.1 企业信息](#21-企业信息)
      - [2.2 求职信息](#22-求职信息)
    - [三、开源社区](#三开源社区)
      - [3.1 社区信息](#31-社区信息)
      - [3.2 开源活动](#32-开源活动)
      - [3.3 开源维护](#33-开源维护)
  - [投稿](#投稿)
  - [致谢](#致谢)

## 已讨论话题索引

### 学术界

| 分类 | 主题 | 细节 | 周刊 |
| --- | --- | ---- | ------ |
| 基础概念 | 任务及其分类 | 从任务分类讲起，进一步细致讨论job、task与instance之间的关系。 | [第2期](./docs/issue-2.md) |
| 基础概念 | 多调度器调度系统 | 介绍现有的多调度器调度系统的框架和案例，并进行总结和对比。 | [第9期](./docs/issue-9.md) |
| 基础概念 | Flash Attention | 主要介绍Flash Attention 1相关内容，并对其后续多个版本进行简单的介绍。 | [第31期](./docs/issue-31.md) |
| 会议分析 | [European Conference on Computer Systems (EuroSys, CCF-A)](https://2025.eurosys.org/) | 对OS领域CCF-A类会议EuroSys进行介绍，从基本信息、投稿难度、侧重领域等基本情况分析，到重点领域分析、重点论文分析。 | [第3期](./docs/issue-3.md) |
| 前沿论文 | [云原生数据库综述 Cloud-Native Databases(TKDE'24，CCF-A)](https://ieeexplore.ieee.org/document/10574374) | 从云原生数据库的演进讲起，进一步细致分类并分析每类云原生数据库的特点与问题，最终讨论可能的未来研究方向。 | [第1期](./docs/issue-1.md) |
| 前沿论文-数据 | [云原生数据库-Amazon Aurora（SIGMOD'17，CCF-A）](https://dl.acm.org/doi/10.1145/3035918.3056101) | 基于第1期进一步介绍案例，介绍云原生数据库 Amazon Aurora。 | [第6期](./docs/issue-6.md) |
| 前沿论文-网络 | [大模型训练网络-RDMA over Ethernet for Distributed AI Training at Meta Scale（SIGCOMM'24，CCF-A）](https://dl.acm.org/doi/10.1145/3035918.3056101) | 介绍 Meta 公司在分布式大模型训练下的网络管理技术。 | [第7期](./docs/issue-7.md) |
| 前沿论文-计算 | [大模型训练容错- Minder: Faulty Machine Detection for Large-scale Distributed Model Training（NSDI'25，CCF-A）](https://www.usenix.org/conference/nsdi25/presentation/deng) | 通过字节的数据介绍了大模型训练场景下的机器故障的检测及诊断方式。 | [第11期](./docs/issue-11.md) |
| 前沿论文-计算 | [算力网络综述 Computing Power Network: A Survey （China Communications，JCR分区Q2，中科院分区Q4）](https://ieeexplore.ieee.org/document/10495806) | 介绍了计算能力网络（算力网络，CPN）。对算力网络的最新研究成果进行了详尽回顾，首先概述了算力网络，再全面阐述了算力建模、信息感知与发布、资源分配、网络转发、算力交易平台和资源协调平台等问题，建立并评估了算力网络测试平台，讨论了算力网络的应用和用例。 | [第13期](./docs/issue-13.md) |
| 前沿论文-计算 | [跨域Spot实例支持大模型 SkyServe: Serving AI Models across Regions and Clouds with Spot Instances (EuroSys'25，CCF-A)](https://dl.acm.org/doi/10.1145/3689031.3717459) | 介绍基于非侵入多云SkyComputing技术的应用，具体面向AI模型推理服务，利用跨Region、跨Cloud下的Spot实例降低成本的同时保障性能（p99延迟）。 | [第14期](./docs/issue-14.md) |
| 前沿论文-计算 | [vCPU抽象和任务调度优化 Optimizing Task Scheduling in Cloud VMs with Accurate vCPU Abstraction](https://example.com/paper) | 介绍vSched系统，通过在虚拟机内部探测vCPU的动态特性（容量、拓扑、活动性），在无需修改Hypervisor的前提下，实现准确的vCPU抽象并优化任务调度。 | [第15期](./docs/issue-15.md) |
| 前沿论文-计算 | [考虑干扰的任务共置成本优化 Eva: Cost-Efficient Cloud-Based Cluster Scheduling (EuroSys'25，CCF-A)](https://arxiv.org/pdf/2503.07437v1) | 介绍在基于云环境的集群配置问题，考虑任务共置会对性能产生干扰的前提下，选择合适的任务共置以实现最优化成本。 | [第16期](./docs/issue-16.md) |
| 前沿论文-计算 | [碎片整理RL重调度 Towards VM Rescheduling Optimization Through Deep Reinforcement Learning (EuroSys'25，CCF-A)](https://dl.acm.org/doi/10.1145/3689031.3717476) | 介绍RL在VM重调度（集群内）问题的应用，引出了许多业界实践经验，开源了代码及数据集。| [第17期](./docs/issue-17.md) |
| 前沿论文-计算 | 可靠性评估/优化部署领域概览及代表论文 | 介绍可靠性评估、可靠性优化部署领域近几年论文情况，选择网络、云、边三个场景下的三篇代表论文进行介绍（分别发表在INFOCOM、TCC、TMC顶会顶刊）。| [第30期](./docs/issue-30.md) |

### 工业界 or 实践

| 分类 | 主题 | 细节 | 周刊 |
| --- | --- | ---- | ------ |
| 基础概念 | 云服务分类 | 对云服务的分类进行介绍，从计算产品、存储、网络、数据库等方面进行分类。 | [第8期](./docs/issue-8.md) |
| 基础概念 | 如何从零构建大模型 | 介绍了如何从零构建大模型的数据处理、Attention机制、GPT模型及模型训练 | [第18期](./docs/issue-18.md) |
| 求职信息 | 云计算领域就业情况 | 对云计算领域就业情况进行介绍，从相关岗位、今年就业情况等基本情况分析，到找工作时间线规划、准备工作规划。 | [第4期](./docs/issue-4.md) |
| 业界现状 | 大模型的技术概览 | 对于大模型的原理，技术方向，行业内的大模型做出介绍 | [第5期](./docs/issue-5.md) |
| 业界现状 | 容错方案演进 | 对容错相关定义、架构以及单元化进行介绍，从单机房到多机房、从单地域到多地域、从数据备份到多活等方面进行介绍。 | [第10期](./docs/issue-10.md) |
| 业界现状 | AI-Infra框架 | 从为何需要AI-Infra框架说起，对AI-Infra框架的演进脉络与框架横向对比进行介绍，最后对典型代表vLLM论文进行介绍。 | [第24期](./docs/issue-24.md) |
| 业界现状 | 机器学习平台 | 对于业内的机器学习平台的内容、开源相关项目、求职相关内容做出介绍 | [第25期](./docs/issue-25.md) |


**[⬆ back to ToC](#目录)**


## 待讨论话题清单

### 一、学术界

#### 1.1 前沿论文
* 在云计算领域，最近有哪些值得关注的前沿研究？
* 这些研究主要关注哪些问题？提出了哪些创新的解决方案？是否有深刻的洞见或启示💡？
* 这些论文的关键结论是什么？有哪些重要的实验结果值得关注？

#### 1.2 顶级期刊
* 哪些期刊在云计算领域中具有重要影响力？
* 这些期刊的发表周期是怎样的？覆盖了哪些细分领域？
* 最近有哪些发表的论文值得特别关注？

**待讨论期刊清单**:
- [IEEE Transactions on Parallel and Distributed Systems (TPDS, CCF-A)](https://www.computer.org/csdl/journal/td)
- [IEEE Transactions on Service Computing (TSC, CCF-A)](https://www.computer.org/csdl/journal/sc)
- [IEEE Transactions on Cloud Computing (TCC, CCF-C)](https://www.computer.org/csdl/journal/cc)

**[⬆ back to ToC](#目录)**

#### 1.3 顶级会议
* 在云计算领域中，有哪些会议是最具影响力的？
* 这些会议的发表周期是怎样的？涵盖了哪些研究方向？
* 最近的会议有哪些新趋势和重要的研究成果？

**待讨论话题**：
- [USENIX Symposium on Operating Systems Design and Implementation (OSDI, CCF-A)](https://www.usenix.org/conference/osdi25)
- [Symposium on Operating Systems Principles (SOSP, CCF-A)](https://sigops.org/s/conferences/sosp/2024/)
- [USENIX Annual Technical Conference (ATC, CCF-A)](https://www.usenix.org/conference/atc25)
- [International World Wide Web Conference (WWW, CCF-A)](https://www2025.thewebconf.org)
- [ACM Symposium on Cloud Computing (SOCC, CCF-B)](https://acmsocc.org/2024/)
- [IEEE International Conference on Cluster Computing (CLUSTER, CCF-B)](https://clustercomp.org/2025/)
- [IEEE International Conference on Distributed Computing Systems (ICDCS, CCF-B)](https://icdcs2025.icdcs.org)
- [International Conference on Parallel Processing (ICPP, CCF-B)](https://icpp2024.org)
- [International European Conference on Parallel and Distributed Computing (Euro-Par, CCF-B)](https://2025.euro-par.org)
- [IEEE World Congress on SERVICES (ICWS, CCF-B)](https://services.conferences.computer.org/2025/icws-2025/)
- [IEEE INTERNATIONAL CONFERENCE ON CLOUD COMPUTING (IEEE Cloud, CCF-C)](https://services.conferences.computer.org/2025/cloud/)
- [USENIX Symposium on Networked Systems (NSDI, CCF-A)](https://www.usenix.org/conference/nsdi25)
- [IEEE International Conference on Computer Communications (INFOCOM, CCF-A)](https://infocom2025.ieee-infocom.org)

**[⬆ back to ToC](#目录)**

#### 1.4 高校团队
* 哪些高校在云计算领域中有重要的研究人员或团队？
* 这些高校的研究方向在云计算领域内的重点是什么？

### 二、工业界

#### 2.1 企业信息
* 哪些企业在云计算领域占据了领先地位？
* 这些企业内部有哪些部门专注于云计算？部门的架构是怎样的？
* 在这些企业中，有哪些知名人物（领导者、行业影响者）？
* 这些企业目前的研究重点或核心项目是什么？

#### 2.2 求职信息
* 云计算相关的公司提供哪些类型的岗位？
* 这些公司对于求职者有哪些具体要求？重点技能有哪些？

### 三、开源社区

#### 3.1 社区信息
* 如何找到适合自己的项目或社区？
* 在云计算领域，哪些开源社区具有重要影响力？
* 有哪些值得关注的开源活动可以参与或跟进？  
  **待讨论话题**：KubeCon、阿里云栖大会。
* 这些开源社区中，有哪些项目值得深入研究或参与贡献？

#### 3.2 开源活动
* 有哪些云计算相关的竞赛值得关注和参与？
* 有哪些开源活动适合云计算领域的研究者和开发者？  
  **待讨论话题**：Google Summer of Code (GSOC)、Open Source Promotion Plan (OSPP)。

#### 3.3 开源维护

* 开源社区规范
  * 开源社区中通常有哪些行为准则？为什么会提出这些准则？这些准则对于社区和贡献者意味着什么？
  * 贡献者如何解读社区中常见的技术文档？如何快速理解社区的规则和文化？
*	参与开源活动
	 *	在开源领域中，云计算领域有哪些值得关注的热门项目？
	 *	参与开源项目有哪些常见方式？除代码提交外，还有哪些途径能够对社区作出贡献？
	 *	提交代码时，如何撰写高质量的Pull Request，使之更容易被社区接受？

**待讨论话题**：
- [GitHub 开源贡献指南](https://docs.github.com/cn/get-started/quickstart/contributing-to-projects)
- [如何撰写优秀的Pull Request](https://github.com/kubernetes/community/blob/master/contributors/guide/pull-requests.md)

**[⬆ back to ToC](#目录)**

## 投稿

欢迎大家积极投稿！请遵循我们的 [投稿指南](./docs/contributing.md)，以确保内容的统一性和高质量。

## 致谢

灵感来源于 [Awesome-LLMOps](https://github.com/tensorchord/Awesome-LLMOps)，特别致敬！🫡


**[⬆ back to ToC](#目录)**
