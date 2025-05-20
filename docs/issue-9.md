![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%290.jpg)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%291.png)

_Eva: Cost\-Efficient Cloud\-Based Cluster Scheduling_

<span style="color:#FF0000"> __EuroSys __ </span>  <span style="color:#FF0000"> __\(CCF\-A\)__ </span>  <span style="color:#FF0000"> __ __ </span>  <span style="color:#FF0000"> __20__ </span>  <span style="color:#FF0000"> __2__ </span>  <span style="color:#FF0000"> __5__ </span>

<span style="color:#A29C71"> __汇报人：樊明晨__ </span>  <span style="color:#A29C71"> __      __ </span>  <span style="color:#A29C71"> __导师：__ </span>  <span style="color:#A29C71"> __丁志军__ </span>  <span style="color:#A29C71"> __ __ </span>

<span style="color:#A29C71"> __专业：计算机科学与技术    __ </span>  <span style="color:#A29C71"> __ __ </span>

<span style="color:#A29C71"> __学院：计算机学院__ </span>

批处理工作负载（如 ML 训练）在研究和企业生产环境中越来越普遍，主流方案是将这些工作负载托管在 <span style="color:#FF0000"> __专用的__ </span> 、 <span style="color:#FF0000"> __固定大小__ </span> 的集群上。

为了满足 ML 训练的需求，许多专门用于ML工作类型的调度程序被设计出来，这些调度器专注于高效利用昂贵的加速器（如 GPU）。

调度作业以最小化 JCT 并最大限度地提高资源利用率。

有研究表明，集群内大数据和 ML的资源请求是突发的，并且会随着时间的变化而波动，这会导致在固定大小的集群中，资源的利用率不足和使用效率低。

云计算可以按需动态扩展和 <span style="color:#FF0000"> __调整__ </span> 计算资源，即用即付模式为使用基于云的集群托管批处理工作负载提供了机会，许多公司已将批处理作业计算迁移到云中。

资源的按需预置消除了作业因资源不足而在队列中等待的时间，这曾是大多数固定大小的集群调度器的主要关注点。

虽然在固定大小的集群设置中，批处理工作负载的任务调度已经被广泛研究，但基于云的集群中的额外灵活性增加了调度问题的复杂性。

这些因素将调度问题的目标从仅最小化作业完成时间 （JCT） 转变为 <span style="color:#FF0000"> __最小化总配置成本__ </span> ，并保持作业吞吐量不变。由于 <span style="color:#FF0000"> __任务调度__ </span> 和 <span style="color:#FF0000"> __实例配置__ </span> 从根本上是相互关联的，因此应安排任务以有效利用实例中的可用资源，而配置实例要匹配任务的需求，因此应共同优化这两个方面以确定最佳配置，其中包括组成集群的实例数量和类型以及任务到实例的分配。

基于云的集群可以通过利用云提供商提供的各种异构实例来动态调整其组成，每种实例类型的成本都不同。

  * 先前的研究表明，在生产环境中，不同任务的资源需求之间没有关联，这意味着可以将具有互补资源需求的任务放在一起，以减少空闲资源的数量。
  * 固定大小的集群调度器（如 Tetris和 Synergy ）使用调度算法，利用 <span style="color:#FF0000"> __任务互补__ </span> 来减少作业排队延迟。
  * 在基于云的集群环境中，任务合作可改善资源分配并减少需要的实例数量，从而降低总成本，然而，大多数现有的云调度器并没有考虑到这一点。

批处理作业表现出不同的资源需求。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%292.png)

  * 之前的工作将共址干扰的影响纳入集群调度器中。Paragon 和 Quasar 使用协同滤波来估计干扰对任务的影响，而 Owl则直接预先分析干扰。根据这些信息，他们的调度程序避免了可能导致彼此严重干扰的任务放在一起，从而满足用户指定的服务质量。
  * 之前的调度算法中不直接考虑 __预置成本。__

位于同一个云实例上的任务要共享低级资源，例如LLC、磁盘 I/O 带宽和网络带宽，共享资源争用会导致共存作业之间的干扰，从而导致性能下降，性能的下降可能会增加作业持续时间，从而增加实例正常运行时间，从而导致更高的总成本。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%293.png)

  * 现有的云调度器，如 Stratus 尽可能避免任务迁移。但是，成本的降低可以累积起来，从而节省大量成本，尤其是对于长时间运行的批处理工作负载。
  * 在这种情况下，保守的迁移策略是次要的。
  * 更重要的是要采用定量方法来评估迁移开销和在集群重新配置节省的成本，在二者之间权衡，从而最大限度地降低总成本

当作业提交到系统或在系统中完成时， <span style="color:#FF0000"> __最佳（__ </span>  <span style="color:#FF0000"> __最大限度地降低成本__ </span>  <span style="color:#FF0000"> __）__ </span> 集群配置可能会随着时间的推移而变化，因此有必要重新配置集群，但这些重新配置可能会产生不可忽略的迁移开销。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%294.png)

__设计一个启发式算法（借鉴__  __VSBPP__  __）并__  __提出__  __ reservation price__  __概念__

__ __  __提出__  __throughput\-normalized__

__ reservation price__  __概念__

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%295.png)

从一维的 VSBPP 的有效启发式方法中获得灵感。

启发式方法首先考虑最大的bin类型，然后反复用适合的最大球填充当前 bin。当无法容纳更多球时，将打开一个相同类型的新bin。如果 bin 中的球可以放入较小的 bin 类型，则启发式将切换到下一个最大的 bin 类型并重复该过程。

从较大的 bin 类型开始会增加多个球打包到一个较大的 bin 中的可能性，从而降低总成本。同样，以降序排列球可以最大限度地减少 bin 中未使用的空间或碎片。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%296.png)

__ Reservation Price__

在多维配置中，“size”的概念变得不太适用，因为资源类型有多种，为了让启发式算法适用，我们需要寻找“size”的替代指标。

对于instance\,由于实例类型的每小时成本与其拥有的资源数量和类型成正比，因此我们可以用每小时成本评估实例类型。

对于task，一个适用的概念是 <span style="color:#FF0000"> __Reservation Price__ </span> 。在 Eva 中，定义为能够满足任务资源需求的最便宜的实例类型的每小时成本。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%297.png)

__ Full Reconfiguration__

根据预留价格，设计 Full Reconfiguration 算法，它考虑了当前系统中用于 reconfiguration 的所有任务。

该算法按 <span style="color:#FF0000"> __成本降序__ </span> 迭代所有可用的实例类型（第 2 行）。这会优先考成本更高（如 GPU）的实例类型，以最大限度地减少成本高昂的资源碎片。对于每种实例类型，该算法会反复尝试预置新实例（第 4\-19 行）。每次都选择使 T ∪ \{τ\} 的总预留价格最大化的未分配任务 τ（第 8 行）。如果添加 τ 导致总预留价格降低，则算法将停止添加任务（第 9\-11 行）。否则，τ 被添加到 T （第 12 行），并且该过程将继续，直到没有更多任务可以打包到实例上。然后，该算法检查将 T 分配给当前实例是否具有成本效益（第 14 行）。如果是，则具有其分配任务 T 的实例将添加到新配置中（第 15 行），并且算法会尝试再次提供相同实例类型的另一个实例。如果没有，算法将继续处理下一个更便宜的实例类型（第 17 行）并重复该过程。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%298.png)

__ Full Reconfiguration__

example：

算法复杂度：

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%299.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2910.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2911.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2912.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2913.png)

__ Throughput\-Normalized Reservation Price__

__用__  <span style="color:#FF0000"> __Co\-location Throughput Table__ </span>  __数据结构维护__

为了解决共存任务之间的干扰导致的 <span style="color:#FF0000"> __性能下降__ </span> ，并降低成本，扩展出这一概念。

具体来说，如果将一组任务 T 分配给实例导致任务 τ ∈ T 具有标准化吞吐量 tputτ，则Throughput\-Normalized Reservation Price表示为                      =

为了便于讨论，将一组任务的吞吐量标准化预留价格定义为

如果任务集的Throughput\-Normalized Reservation Price超过实例的实际成本，则认为任务到实例的分配具有成本效益。

预先构造表成本很高，Eva 通过从任务中观察到的吞吐量更新表。在查找一组共置任务T的共置吞吐量时，如果之前已经观察到 T 并且已经记录在表中，则取此值，如果不是，它将 τ 的吞吐量估计为                         ，

如果尚未记录 tputτ，τ′ ，则使用默认值 t 初始化，该默认值是 Eva 的可调参数，t越小，共置时就会越保守。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2914.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2915.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2916.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2917.png)

__ Throughput\-Normalized Reservation Price__

多任务作业在批处理中很普遍，在这些情况下，来自同一作业 j 的各个任务的执行可能是相互依赖的。

比如 j 中的一个任务由于来自共置的干扰而出现性能下降，则 j 中的所有任务的吞吐量都会受到影响。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2918.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2919.png)

__Partial Reconfiguration__  __（部分重配置）__

Full Reconfiguration算法考虑 __所有__ 任务，它不考虑当前集群配置，这可能会导致任务迁移过多以及实例频繁启动或终止。

为了缓解这种情况，我们引入 <span style="color:#FF0000"> __Partial Reconfiguration__ </span>  算法，仅考虑重新配置的任务子集，而集群配置的其余部分保持不变。

__实质：限定任务子集__  __\+__  __Full Reconfiguration__  __算法__

partial Reconfiguration  <span style="color:#FF0000">vs</span>  FullReconfiguration

__P__

Full算法会产生大量的迁移开销，而 Partial算法则无法实现最优配置。

Eva 采用了集成方法，在每个调度周期，Eva 都会运行两种算法来获得两种配置，并决定采用哪一种。

本文提出了一个定量标准来决定 使用哪种算法。

设 SF （SP） 是完全（部分）重新配置的单位时间预置成本节省大小。

设 MF （MP） 为 迁移成本。

D是一次配置维持的时间。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2920.png)

包含多个任务的作业可以提交到eva

如果没有给定吞吐量，可以使用  <span style="color:#FF0000"> __Profiler __ </span> 估计吞吐量

Eva 执行定期调度\,在每个调度周期结束时（例如 5 分钟）， <span style="color:#FF0000"> __Scheduler __ </span> 会确定集群配置，包括实例数、每个实例的类型以及向实例分配任务。

根据配置， <span style="color:#FF0000"> __Provisioner__ </span>  从云提供商启动和终止实例，而  <span style="color:#FF0000"> __Executor__ </span>  在这些实例上启动和迁移任务。

通过分析来获得任务共置对吞吐量的影响成本高， <span style="color:#FF0000"> __ThroughputMonitor__ </span>  会在线跟踪和学习吞吐量的变化，维护共置吞吐量表——一个记录共置任务吞吐量的数据结构。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2921.png)

_实现：_

在 Python 中实现了 Eva和一个simulator。

Eva 遵循模块化架构以实现可扩展性，并采用集中式 Master\-Worker 模型。

Master 通过现有的云平台 API 管理云实例。实例实例化后，将在实例上启动一个 worker，该 worker 通过 gRPC 与主服务器通信。

要使用 Eva，用户只需 提供一个 Dockerfile 及其执行工件并指定所需的资源，类似于现有的基于容器的云平台，如Google Ku bernetes Engine。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2922.png)

<span style="color:#005692"> __4\. __ </span>  <span style="color:#005692"> __怎么样__ </span>  <span style="color:#005692"> __\-__ </span>  <span style="color:#005692"> __基于__ </span>  <span style="color:#005692"> __AWS EC2__ </span>  <span style="color:#005692"> __的物理实验__ </span>

_实验设置：_

考虑AWS EC2 上来自 3 个系列的21种实例类型：P3 实例（GPU实例）、C7i 实例（计算优化实例）和 R7i 实例（内存优化实例）。

考虑了来自各种 ML 和科学计算应用程序的 10 种不同的批处理工作负载。分为大小规模，小规模跟踪32个作业，大规模跟踪120个作业。

作业到达时间根据泊松到达过程生成，平均到达间隔时间为 20 分钟。

_对比算法：_

No\-Packing Scheduler（每个任务都托管在单独实例）

Stratus（尽可能减小迁移开销）

Owl（尽可能减少共置干扰）

Synergy（尽可能减少碎片资源）

_选取指标：_

<span style="color:#FF0000"> __成本__ </span>

资源分配（已分配资源与总资源的比率）

JCT

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2923.png)

<span style="color:#005692"> __4\. __ </span>  <span style="color:#005692"> __怎么样__ </span>  <span style="color:#005692"> __\-__ </span>  <span style="color:#005692"> __基于__ </span>  <span style="color:#005692"> __AWS EC2__ </span>  <span style="color:#005692"> __的物理实验__ </span>  <span style="color:#005692"> __ __ </span>

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2924.png)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2925.png)

_实验设置：_

使用了阿里巴巴中公开可用的生产跟踪 （cluster\-trace\-gpu\-v2023）

原始跟踪仅包含单任务作业，在模拟实验中将每个任务视为单个任务作业。

原始跟踪中包括高比例的短作业，其中 80% 持续时间不到一小时，半程持续时间少于 11 分钟。

为了更好地表示 MLtraining 作业的长期运行性质，还在单独的实验中使用了 Gavel对作业持续时间建模。

_对比算法：_

No\-Packing Scheduler（每个任务都托管在单独实例）

Stratus（尽可能减小迁移开销）

Owl（尽可能减少共置干扰）

Synergy（尽可能减少碎片资源）

_选取指标：_

成本

资源分配（已分配资源与总资源的比率）

JCT

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2926.png)

Gavel对作业持续时间建模跟踪结果

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2927.png)

本文仅是对批处理负载做的工作，可以考虑更多的负载类型，让调度器可以工作在多种任务类型的环境中。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2928.png)

从实验结果可以看出，JCT比基线要大，但我们的出发点是尽可能减少成本，所以在本次实验中这是可以容忍的，也是后续优化的一个方向。

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2929.jpg)

![](img%5CEva%28%E6%A8%8A%E6%98%8E%E6%99%A8%2930.png)

<span style="color:#A29C71"> __汇报人：樊明晨__ </span>  <span style="color:#A29C71"> __      __ </span>  <span style="color:#A29C71"> __导师：__ </span>  <span style="color:#A29C71"> __丁志军__ </span>  <span style="color:#A29C71"> __ __ </span>

<span style="color:#A29C71"> __专业：计算机科学与技术    __ </span>  <span style="color:#A29C71"> __ __ </span>

<span style="color:#A29C71"> __学院：计算机学院__ </span>

