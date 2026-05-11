from typing import Any, Dict, List, Union, Tuple 
from pathlib import Path  
from dataclasses import asdict  # dataclass 工具：把 PDM 打分返回的 dataclass 转成普通 dict，方便写入 DataFrame。
from datetime import datetime  
import traceback  # 异常调试工具：某个 token 评估失败时打印完整堆栈。
import logging  
import lzma  # 压缩文件工具：NAVSIM 的 metric cache 文件使用 lzma 压缩。
import pickle  
import os  
import uuid  # 唯一 ID 工具：给每个 worker 生成 thread_id，方便在日志中区分并行任务。

import hydra  # 配置框架：读取 YAML 配置，并把命令行覆盖项合并成 cfg。
from hydra.utils import instantiate  # Hydra 实例化工具：根据配置里的 _target_ 动态构造 Python 对象。
from omegaconf import DictConfig  # Hydra/OmegaConf 的配置对象类型。
import pandas as pd  

from nuplan.planning.script.builders.logging_builder import build_logger  # nuPlan API：根据 cfg 初始化日志系统。
from nuplan.planning.utils.multithreading.worker_utils import worker_map  # nuPlan API：把任务分发到顺序/并行 worker 上执行。

from navsim.agents.abstract_agent import AbstractAgent  # NAVSIM agent 接口：所有规划器都需要符合这个接口。
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader  # NAVSIM 数据加载：场景、筛选器、metric cache。
from navsim.common.dataclasses import SensorConfig  # NAVSIM 传感器配置：声明是否加载相机、LiDAR 等输入。
from navsim.evaluate.pdm_score import pdm_score  # NAVSIM 评估入口：根据预测轨迹计算 PDM score。
from navsim.planning.script.builders.worker_pool_builder import build_worker  # NAVSIM worker 构建器：根据 cfg.worker 创建 worker pool。
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # PDM 仿真器：用于非反应式仿真。
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer  # PDM 评分器：计算 NC/DAC/TTC/comfort/EP 等指标。
from navsim.planning.metric_caching.metric_cache import MetricCache  # MetricCache 类型：表示预计算的评估缓存。

logger = logging.getLogger(__name__)  # 当前文件的 logger；真正的格式和输出位置由 build_logger(cfg) 设置。

CONFIG_PATH = "config/pdm_scoring"  # Hydra 配置目录；相对 navsim/planning/script 目录查找。
CONFIG_NAME = "default_run_pdm_score"  # Hydra 默认配置名；对应 default_run_pdm_score.yaml。


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    在一个 worker 内执行一批场景的 PDMS 评估。

    main() 会先扫描 navtest 中有哪些 token 可以评估，再通过 worker_map()
    将 token 按 log 分组后分发给这个函数。这个函数内部会实例化 simulator、
    scorer 和 agent，加载对应的传感器输入与 metric cache，调用 agent 生成轨迹，
    最后调用 pdm_score() 得到每个 token 的指标。

    :param args: worker_map 传入的一组任务，每个元素包含 cfg、log_file、tokens。
    :return: 当前 worker 负责的所有 token 的评估结果列表。
    """
    node_id = int(os.environ.get("NODE_RANK", 0))  # 读取多机训练/评估中的节点编号；单机运行时默认为 0。
    thread_id = str(uuid.uuid4())  # 给当前 worker 生成一个唯一 id，用于日志追踪。
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}") 

    log_names = [a["log_file"] for a in args]  # 当前 worker 被分配到的 log 文件名列表。
    tokens = [t for a in args for t in a["tokens"]]  # 当前 worker 被分配到的所有场景 token。
    cfg: DictConfig = args[0]["cfg"]  # 取出 Hydra 配置；同一个 worker 的所有任务共用同一份 cfg。

    simulator: PDMSimulator = instantiate(cfg.simulator)  # 根据 cfg.simulator 的 _target_ 创建 PDM simulator。
    scorer: PDMScorer = instantiate(cfg.scorer)  # 根据 cfg.scorer 的 _target_ 创建 PDM scorer。
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"  # simulator 和 scorer 的轨迹采样参数必须一致，否则时间轴无法对齐。
    agent: AbstractAgent = instantiate(cfg.agent)  # 根据 cfg.agent 创建 agent；命令行 agent=diffusiondrive_agent 会在这里实例化 DiffusionDrive。
    agent.initialize()  # 初始化 agent；对 DiffusionDrive 来说，主要是在这里加载 agent.checkpoint_path 指定的权重。

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))  # 加载 metric cache 的索引，后续可通过 token 找到 cache 文件路径。
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)  # 创建当前 split 的场景筛选器，例如 navtest 的 token/log 过滤规则。
    scene_filter.log_names = log_names  # 限制当前 worker 只读取分配给它的 log。
    scene_filter.tokens = tokens  # 限制当前 worker 只读取分配给它的 token。
    scene_loader = SceneLoader(  # 创建真正用于推理的 SceneLoader；这里会按 agent 的需求加载传感器数据。
        sensor_blobs_path=Path(cfg.sensor_blobs_path),  # 传感器数据目录，例如相机图像、LiDAR blob。
        data_path=Path(cfg.navsim_log_path),  # NAVSIM log annotation 数据目录。
        scene_filter=scene_filter,  # 当前 worker 的场景筛选范围。
        sensor_config=agent.get_sensor_config(),  # 由 agent 声明需要哪些传感器；DiffusionDrive 在这里决定加载哪些输入。
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))  # 只评估“场景存在且 metric cache 也存在”的 token。
    pdm_results: List[Dict[str, Any]] = []  # 保存当前 worker 的所有 token 评估结果。
    for idx, (token) in enumerate(tokens_to_evaluate):  # 逐个处理当前 worker 分配到的 token。
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )  # 打印当前 token 的处理进度。
        score_row: Dict[str, Any] = {"token": token, "valid": True}  # 初始化当前 token 的结果行；默认该场景有效。
        try:  # 单个 token 失败时只标记该 token，不中断整个评估。
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]  # 根据 token 找到对应的 metric cache 文件。
            with lzma.open(metric_cache_path, "rb") as f:  # 以二进制方式打开 lzma 压缩 cache 文件。
                metric_cache: MetricCache = pickle.load(f)  # 反序列化出 MetricCache 对象。

            agent_input = scene_loader.get_agent_input_from_token(token)  # 加载当前 token 的 agent 输入，包括历史状态和所需传感器数据。
            if agent.requires_scene:  # 某些 agent 需要完整 scene 对象；先检查 agent 的接口声明。
                scene = scene_loader.get_scene_from_token(token)  # 如果需要，就额外加载完整 scene。
                trajectory = agent.compute_trajectory(agent_input, scene)  # 调用 agent 生成轨迹；需要 scene 的 agent 走这个分支。
            else:
                trajectory = agent.compute_trajectory(agent_input)  # 调用 agent 生成轨迹；DiffusionDrive 通常走这个分支。

            pdm_result = pdm_score(  # 调用 PDM 打分函数，对当前 token 的预测轨迹进行评估。
                metric_cache=metric_cache,  # 预计算评估缓存，包含地图、路线、目标等评分所需信息。
                model_trajectory=trajectory,  # agent 输出的未来自车轨迹。
                future_sampling=simulator.proposal_sampling,  # 未来轨迹采样设置，必须和 simulator/scorer 保持一致。
                simulator=simulator,  # PDM simulator，用于执行非反应式仿真。
                scorer=scorer,  # PDM scorer，用于计算各项指标。
            )
            score_row.update(asdict(pdm_result))  # 将 PDM 结果转成 dict，并合并到当前 token 的结果行。
        except Exception as e:  # 捕获当前 token 的异常，避免整个 navtest 评估中断。
            logger.warning(f"----------- Agent failed for token {token}:")  # 记录失败 token。
            traceback.print_exc()  # 打印完整异常堆栈，方便定位问题。
            score_row["valid"] = False  # 标记当前 token 评估失败。

        pdm_results.append(score_row)  # 将当前 token 的结果加入 worker 结果列表。
    return pdm_results  # 返回当前 worker 处理完的所有结果。


# 运行这个脚本时，Hydra 会先加载：
# DiffusionDrive/navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml
# 然后把 YAML 配置和命令行覆盖项合并成 cfg，再传给 main(cfg)。
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    PDMS 评估主入口。

    这个函数运行在主进程中，负责初始化日志和 worker，扫描可评估 token，
    将任务分发给 run_pdm_score()，最后汇总并保存 csv。

    :param cfg: Hydra/OmegaConf 合并后的完整配置。
    """

    build_logger(cfg)  # 初始化日志系统；来自 nuPlan，决定日志格式、等级、输出目录等。
    worker = build_worker(cfg)  # 根据 cfg.worker 创建 worker pool；例如 sequential、ray_distributed。

    # 主进程先创建一个轻量 SceneLoader，只用于扫描 token 列表。
    # 这里不做模型推理，也不需要相机/LiDAR 数据，所以 sensor_blobs_path=None。
    # TODO: 原作者说明：未来可从 metadata 推断每个 log 的 token，避免这里同时加载 scene 和 metric cache 索引。
    scene_loader = SceneLoader(
        sensor_blobs_path=None,  # 主进程只扫描 token，不读取传感器 blob。
        data_path=Path(cfg.navsim_log_path),  # NAVSIM log annotation 路径。
        scene_filter=instantiate(cfg.train_test_split.scene_filter),  # 根据 navtest/navtrain 等 split 配置创建筛选器。
        sensor_config=SensorConfig.build_no_sensors(),  # 不加载任何传感器数据，只取 token。
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))  # 加载 metric cache 索引，用来判断哪些 token 有评分缓存。

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))  # 只有“场景存在”且“metric cache 存在”的 token 才会被评估。

    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))  # 场景存在但缺少 metric cache 的 token 数。
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))  # metric cache 存在但当前 split 不使用的 token 数。
    if num_missing_metric_cache_tokens > 0:  # 如果有场景缺少 metric cache，发出警告并跳过。
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:  # 如果有多余 cache，发出警告但不影响评估。
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))  # 记录即将评估的场景数量。
    data_points = [  # 准备分发给 worker 的任务列表。
        {
            "cfg": cfg,  # 把完整配置传给 worker；worker 内部还要实例化 agent/simulator/scorer。
            "log_file": log_file,  # 当前任务对应的 log 文件名。
            "tokens": tokens_list,  # 当前 log 下的 token 列表。
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()  # 按 log 对 token 分组，便于 worker 分发。
    ]
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)  # 通过 worker pool 调用 run_pdm_score，并收集所有结果。

    pdm_score_df = pd.DataFrame(score_rows)  # 把每个 token 的评分结果转换成表格。
    num_sucessful_scenarios = pdm_score_df["valid"].sum()  # 统计成功评估的场景数量；变量名沿用原代码拼写。
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios  # 统计失败场景数量。
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)  # 对所有数值指标求平均，得到最终平均行。
    average_row["token"] = "average"  # 平均行的 token 字段标记为 average。
    average_row["valid"] = pdm_score_df["valid"].all()  # 只有所有 token 都有效时，平均行 valid 才为 True。
    pdm_score_df.loc[len(pdm_score_df)] = average_row  # 将平均行追加到表格最后。

    save_path = Path(cfg.output_dir)  # 结果输出目录，通常由 NAVSIM_EXP_ROOT、experiment_name、experiment_uid 组成。
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")  # 用当前时间生成 csv 文件名。
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")  # 保存每个 token 和 average 行的评估结果。

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )  # 评估结束后，在日志中打印汇总信息和结果文件路径。


if __name__ == "__main__":  # 只有直接执行该文件时才运行 main；被其他模块 import 时不会自动执行。
    main()  # 触发 Hydra main：先加载配置，再调用 main(cfg)。
