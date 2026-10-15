# 自动调优

`cutile::tune` 会在一个声明好的 kernel 配置空间里做搜索，用 `cutile::bench` 的设备事件计时测量每个候选，并把试验记录与最终胜出者都持久化下来。它由 `experimental-tune` feature 门控，并且 API 可能在不同版本之间发生变化：

```toml
cutile = { version = "...", features = ["experimental-tune"] }
```

你可以用闭包来调优单个 kernel，也可以实现 `Objective` 来测量整个请求。`autotune` 示例用的是闭包：`cargo run -p cutile-examples --example autotune --features experimental-tune`。

## 声明空间并运行搜索

一个**配置**是一组成员为整数或字符串的具名参数；**空间**则是配置的列表。只有 `configs` 和结尾的 `run` 是必需的。builder 的其余部分都是可选的：`prune` 用来过滤候选，`budget` 限制墙钟时间，`log` 让这次运行可恢复，`require` 声明搜索必须覆盖的现任配置。

```rust
let configs: Vec<Config> = [16i64, 32, 64, 128]
    .into_iter()
    .map(|bn| Config::new([("BN", ParamValue::Int(bn))]))
    .collect();

let output = Autotuner::new("fmha_decode")
    .configs(configs)
    .prune(|c| c.int("BN").unwrap() <= pp)
    .require(vec![shipping_config])
    .budget(Duration::from_secs(300))
    .arch("sm_100") // 记录架构；从另一种架构恢复的日志会被拒绝
    .log("fmha_decode.trials.jsonl")
    .run(&stream, |stream, config| {
        // setup 闭包由你掌控：读取配置，为它构造这个 kernel 的组合启动器，
        // 先跑一次作为正确性闸门，然后返回待计时的闭包。Err(..) 会把该候选
        // 记为无效，搜索继续进行。
        let bn = config.int("BN").unwrap();
        let mut launch = move |stream: &Arc<Stream>| {
            fmha_decode(out.partition([1, bn as i32]), &q, &k, &v)
                .generics(generics_for(bn))
                .sync_on(stream)
        };
        launch(&stream)?;
        Ok(launch)
    })?;
let best: Config = output.best.expect("a winner");
```

两个最好的候选会再进行一次配对的 A/B 决胜测量，以消除时钟与温度漂移的影响。如果入围者在 setup 阶段失败，则直接出局。

每一个被 `require` 的配置都必须属于该搜索空间。这些配置会被**优先**测量，在预算把它们截断之前就完成。最终胜出者必须达到或超过它们中的每一个。

试验日志会记录调优器名称以及搜索空间的哈希。来自不同调优器或不同空间的日志会被拒绝。

<a id="engine-scale-measurement-implements-objective"></a>
## 用 `Objective` 测量整个请求

当目标是端到端指标（每秒 token 数、请求延迟）时，库无法掌控这次启动。这时应改为实现 `Objective`：`configs()` 暴露搜索空间，`measure(index)` 在引擎层面应用一个候选并返回用 `Trial::measured` 或 `Trial::invalid` 构造的 `Trial`，`budget_remaining()` 报告剩余预算。随后 `Autotuner::run_objective` 会提供与闭包路径相同的试验记录、恢复能力以及 `require` 覆盖：

```rust
let mut objective = EngineObjective::new(engine, configs);
let output = Autotuner::new("engine_prefill")
    .arch(current_arch) // 例如 "sm_100"；恢复时会拒绝来自其它架构的日志
    .log("prefill.trials.jsonl")
    .require(vec![current_default])
    .run_objective(&mut objective)?;
```

这条路径没有配对的决胜测量：库无法通过它自己的 bench 闭包去重新计时一个引擎级目标。`Output::best` 就是顺序测量中的最佳中位数。调用方可以把入围者放在一起重新测量，以检查漂移。

序列化后的 `Trial` 记录在 cuTile 升级之间保持兼容。一旦出现不兼容改动，日志的 schema 版本号会递增，`TrialLog::open` 会拒绝旧 schema。

## 固化胜出配置

`tune::Record` 用来持久化胜出配置，让生产环境直接加载它们而不是重新搜索：每个 kernel 一个 JSON 文件，由 `save` 写出，并和被调优的代码放在一起提交。该文件包含一个来源信息头部（kernel 名称、源码哈希、cutile 版本、`tileiras` 指纹、架构、搜索空间哈希），以及每个形状类别桶各一条条目。

```rust
use cutile::tune::{L2Key, Record, RecordEntry, Workspace};

// 调优之后：`winner` 是胜出的 Trial，`best` 是它的 Config
// （即 `output.best`），`launcher` 是用 `best` 的取值构造出的
// 这个 kernel 的组合 builder —— 也就是生产环境派发时用的同一个调用表达式。
let TrialState::Measured { median_ms, reps, .. } = winner.state else {
    unreachable!("the winner was measured");
};
let mut record = Record::new(&workspace);
record.insert(RecordEntry {
    bucket: "tg<=512".into(),
    config: best,
    median_ms,
    samples: reps,
    l2_key: Some(L2Key::current(launcher.l2_cache_key()?)),
});
record.save(&path)?;

// 生产环境：加载，并对照正在运行的工作区做校验。
let (record, warnings) = Record::load_verified(&path, &workspace, |entry| {
    Ok(Some(specialize_for(&entry.config)?.l2_cache_key()?))
})?;
let entry = record.get("tg<=512");
```

**桶（bucket）**是一个普通字符串标签，由 `Record::get` 精确匹配；判定某个运行时形状落入哪个桶的谓词存在于消费者的派发代码里，记录本身只存储这个标签。`median_ms` 和 `samples` 是胜出者的实测中位延迟，以及它背后进行过计时的轮数 —— 这些是给阅读该文件的**人**看的来源信息，而不是校验的输入。`l2_key` 是胜出者的持久化缓存键，由组合 builder 在不编译、不启动的前提下算出（`.l2_cache_key()`）。

如果记录对应的 kernel、架构、源码哈希或搜索空间与当前运行的工作区不同，`load_verified` 会拒绝它。它还会重新计算每条条目的缓存键，以核对该 kernel 的依赖与工具链。只影响计时的改动只会产生警告。

## 在扫描过程中预热与管理 kernel 缓存

对同一个用于 `.execute` 或 `.sync_on` 的 builder 调用 `.compile()`，可以预热将要运行的那个特化。`api::meta` tensor 能在不分配设备内存、不启动 kernel 的前提下提供形状与 dtype：

```rust
my_module::my_kernel(api::meta::<f32>(&[64, 64]).sync()?.partition([16, 16]), ...)
    .generics(generics)
    .compile_options(opts)
    .compile()?;
```

调优扫描和长时运行的推理引擎会累积很多特化，而每个被缓存的 kernel 都占用设备内存。内存中的 kernel 缓存是**有意设计成无上限的**。`cutile::tile_kernel` 里的 `unsafe` 函数 `clear_kernel_cache()`、`evict_kernel(&key)` 和 `retain_kernels(pred)` 无需任何 Cargo feature 即可使用。它们会移除条目，并在每个模块的最后一个持有者被释放时回收其设备内存。这些函数是 `unsafe` 的：在驱逐之前，请同步所有可能仍在运行该缓存 kernel 的流。在两次调优试验之间，或者推理引擎空闲时，调用方可以先驱逐、再重新加载所需的特化。`cutile::tune` 中的自动调优 API 仍然需要 `experimental-tune`。
