# Awesome-Cloud 周刊（第 14 期）：前沿论文-SkyServe: Serving AI Models across Regions and Clouds with Spot Instances


这里简单记录每周分享的前沿内容，不定期发布。

---

## **1. 背景与需求**
- **AI模型推理服务的挑战**  
  - **高成本**：  
    - 依赖昂贵GPU实例（如A10G/T4），处理成本比传统搜索查询高10倍。  
    - 预置副本数量需覆盖流量峰值（峰值可达平均值的50倍）。  
  - **高延迟**：  
    - 单请求处理耗时秒级，但网络延迟占比低，需跨地域部署优化端到端延迟。  
    - 冷启动时间长：实验显示部署Llama-2-7B模型需183秒（AWS预置镜像）。  

- **Spot实例的机遇与问题**  
  - **低成本优势**：价格仅为按需实例的8-50%，利用云厂商闲置算力。  
  - **不稳定性**：  
    - GPU Spot实例中断概率>20%（CPU<5%），抢占警告时间短（AWS 2分钟，GCP/Azure 30秒），远低于服务启动时间。  
  - **静态预置的缺陷**：  
    - AWS ASG固定比例策略在Spot短缺时持续尝试获取Spot实例，导致请求失败率高达36%。  
    - 按需实例冗余造成成本浪费。  

---

## **2. 创新点与解决方案**
- **核心挑战**：  
  1. Spot GPU实例不稳定 + 大模型冷启动时间长 → 服务中断风险高。  
  2. 单一区域Spot资源有限（33.1%时间全区域不可用）且抢占相关性强。  
  3. 静态预置策略无法动态适应资源波动。  

- **SkyServe关键技术**：  
  - **跨区域/云动态部署**：  
    - 评估多区域、多云（AWS/GCP/Azure）的抢占风险，避免高风险区域。  
  - **冗余+动态回退算法（SpotHedge）**：  
    - **冗余放置**：超额配置Spot副本作为缓冲，缓解抢占影响。  
    - **动态回退**：抢占后自动启动按需实例，Spot恢复后缩减按需实例。  
  - **抢占风险评估策略**：  
    - **在线策略**：基于历史抢占记录（维护可用区/高风险区列表）实时决策。  
    - **对比基准**：Omniscient（全知最优，需完整追踪数据） vs. SpotHedge（实际可行）。  

---

## **3. 系统实现（SkyServe架构）**
1. **负载均衡器**：接收用户请求，分发至就绪副本，并转发QPS指标给Autoscaler。  
2. **Autoscaler**：  
   - 集成SpotHedge策略，动态计算目标副本数及超额配置量，生成Spot计划（含回退策略）。  
3. **Service Controller**：  
   - 管理副本生命周期（预置/启动Spot/按需实例），监控健康状态（Readiness Probe），处理抢占事件。  
4. **监控与反馈**：Metric指标用于伸缩决策，形成闭环控制。  

---

## **4. 实验结果**
- **真实实验（AWS, 22小时, 13.3万请求）**：  
  - **模型**：Llama-2-70B（8xA10G）和OPT-6.7B（4xT4）。  
  - **对比基线**：AWS ASG、MArk（适配Spot GPU）、AWSSpot（单区域多可用区）。  
  - **结果**：  
    - **就绪副本稳定性**：SkyServe在Spot高可用（91-100%）和波动（45-46%）场景下均保持稳定，优于基线。  
    - **成本与延迟**：在保证低失败率（<5%）的同时，成本显著低于纯按需实例。  

- **模拟实验（AWS/GCP抢占数据）**：  
  - **工作负载**：泊松分布、真实LLM服务、Azure无服务器调用。  
  - **策略对比**：SpotHedge接近Optimal（ILP），成本节约30%以上，优于Even Spread/Round Robin。  

---

## **5. 未来方向**
1. **延迟敏感优化**：  
   - 动态路由实时请求至客户端同区域副本，保留按需实例池处理短TTFT请求。  
2. **异构GPU支持**：  
   - 智能切换高低性能GPU Spot实例（如A100→T4），平衡成本与性能。  
3. **小模型混合调度**：  
   - 扩展调度策略，权衡小模型的抢占概率与成本。  
4. **资源质量感知**：  
   - 引入网络延迟、算力等指标优化副本放置。  

---

## **总结**
- **核心贡献**：通过跨云动态调度、冗余放置和智能回退，实现低成本、高可用的AI模型推理服务。  
- **效果**：在Spot实例不稳定的环境下，成本降低30%+，服务可用性>95%。  
- **应用场景**：适用于大语言模型、高波动流量的推理服务部署。  

## 问答环节记录

Q. 它是怎么判断高抢占风险区何时变为低抢占风险区的？
A. 利用任在该区域运行的服务实例，或者等待一定时间。


---

Q. 我有个好奇的地方没问，就是在不稳定性那里提到的冷启动时间大于抢占警告时间，这样会有啥后果，没来得及迁的服务会直接丢失吗？
？
A. 对。


---

Q. 1. MArk 为什么不好？CPU 和 GPU 的区别是什么前文没有讲？
    2. 理论最优算法的缺点是什么？在实验中没有体现出来？
    3. 如果ASG 是固定的，那“自动伸缩”体现在哪？
A. 1.Cpu实例被强占频次少
    2. 不能预先知道未来是否被强占
    3. 比例固定，但（自动）买实例


---

Q. 1. 是否有考虑组合各地域的低风险抢占时区
    2.需要多卡运行的推理模型如何在Spot中处理
A. 1.可以考虑。
    2.基于sportserver论文方法


---
Q. 1. 可抢占和不可抢占的时间粒度是否是一样的？
    2. 可抢占实例是会被谁抢占？为什么会被抢占？
    3. 中断概率的含义是什么？（时间的比例）
    4. 贺松和嘉文师兄问题的结合，根据历史信息分析出可用性趋势估计来替代简单的高低风险的估计。
A. 
1. **不完全一样**，但通常以秒级或分钟级为单位。  
- **可抢占实例（Spot Instances）**：  
  - 抢占时间粒度较粗，通常在**秒级到分钟级**。例如，AWS 提供 **2 分钟**的抢占警告，GCP/Azure 仅提供 **30 秒**。  
  - 实际的抢占行为取决于云厂商的调度策略，可能突发性较强。  
- **不可抢占实例（On-demand Instances）**：  
  - 时间粒度更灵活，用户可以按秒、小时或长期租用（如预留实例）。  
  - 除非用户主动终止或付费逾期，否则不会被动中断。  
2. 由**云服务提供商（如 AWS、GCP、Azure）**的**资源调度系统**主动回收。  
3. 中断概率（Preemption Probability）是指 **Spot 实例在运行期间被云厂商强制回收的几率**，通常以**时间比例**或**实例生命周期比例**衡量。  

