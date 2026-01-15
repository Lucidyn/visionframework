"""
跟踪评估示例

本示例展示如何使用 TrackingEvaluator 评估跟踪算法的性能。
包括 MOTA、MOTP、IDF1 等标准跟踪评估指标。

评估指标说明：
- MOTA (Multiple Object Tracking Accuracy): 多目标跟踪精度
- MOTP (Multiple Object Tracking Precision): 多目标跟踪位置精度
- IDF1 (ID F1 Score): 跟踪 ID 保留性能评分
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import TrackingEvaluator


def example_basic_evaluation():
    """
    示例 1: 基本跟踪评估
    
    本示例展示如何使用 TrackingEvaluator 进行基本的跟踪性能评估。
    """
    print("=" * 70)
    print("示例 1: 基本跟踪评估")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化评估器 ==========
    print("\n1. 初始化 TrackingEvaluator...")
    
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    print("  ✓ 评估器初始化成功")
    
    # ========== 步骤 2: 准备示例数据 ==========
    print("\n2. 准备跟踪预测和真值数据...")
    
    # 预测的跟踪结果：3 帧的检测结果，每帧包含跟踪 ID 和边界框
    # 格式：[{frame_0}, {frame_1}, {frame_2}]
    # 每帧包含：[{"track_id": int, "bbox": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}}, ...]
    pred_tracks = [
        # 第 0 帧：2 个预测跟踪
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}  # 左上角的对象
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}  # 右下角的对象
            }
        ],
        # 第 1 帧：2 个预测跟踪（第一个对象移动）
        [
            {
                "track_id": 1,
                "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}  # 跟踪 ID 1 向右下移动
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}  # 跟踪 ID 2 保持不动
            }
        ],
        # 第 2 帧：1 个预测跟踪（漏检）
        [
            {
                "track_id": 1,
                "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}  # 跟踪 ID 1 继续移动
            }
            # 注意：跟踪 ID 2 缺失（漏检）
        ]
    ]
    
    # 真值跟踪数据（地面事实）
    gt_tracks = [
        # 第 0 帧的真值
        [
            {
                "track_id": 1,
                "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}
            }
        ],
        # 第 1 帧的真值
        [
            {
                "track_id": 1,
                "bbox": {"x1": 15, "y1": 15, "x2": 55, "y2": 55}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 100, "y1": 100, "x2": 150, "y2": 150}
            }
        ],
        # 第 2 帧的真值
        [
            {
                "track_id": 1,
                "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}
            },
            {
                "track_id": 2,
                "bbox": {"x1": 105, "y1": 105, "x2": 155, "y2": 155}  # 跟踪 ID 2 移动了
            }
        ]
    ]
    
    print("  ✓ 准备了 3 帧的预测和真值数据")
    
    # ========== 步骤 3: 计算 MOTA ==========
    print("\n3. 计算 MOTA (Multiple Object Tracking Accuracy)...")
    
    mota_result = evaluator.calculate_mota(pred_tracks, gt_tracks)
    
    print("\n  MOTA 评估结果:")
    print(f"    MOTA 得分:      {mota_result['MOTA']:.4f}")
    print(f"    精准度:         {mota_result['precision']:.4f}")
    print(f"    召回率:         {mota_result['recall']:.4f}")
    print(f"    总真值数:       {mota_result['total_gt']}")
    print(f"    误报数:         {mota_result['total_fp']}")
    print(f"    漏报数:         {mota_result['total_fn']}")
    print(f"    ID 切换次数:    {mota_result['total_id_switches']}")
    
    # ========== 步骤 4: 计算 MOTP ==========
    print("\n4. 计算 MOTP (Multiple Object Tracking Precision)...")
    
    motp_result = evaluator.calculate_motp(pred_tracks, gt_tracks)
    
    print("\n  MOTP 评估结果:")
    print(f"    MOTP 得分:      {motp_result['MOTP']:.4f} 像素")
    print(f"    匹配对数:       {motp_result['total_matched_pairs']}")
    
    # ========== 步骤 5: 计算 IDF1 ==========
    print("\n5. 计算 IDF1 (ID F1 Score)...")
    
    idf1_result = evaluator.calculate_idf1(pred_tracks, gt_tracks)
    
    print("\n  IDF1 评估结果:")
    print(f"    IDF1 得分:      {idf1_result['IDF1']:.4f}")
    print(f"    ID TP (IDTP):   {idf1_result['IDTP']}")
    print(f"    ID FP (IDFP):   {idf1_result['IDFP']}")
    print(f"    ID FN (IDFN):   {idf1_result['IDFN']}")
    
    # ========== 步骤 6: 综合评估 ==========
    print("\n6. 综合评估结果...")
    
    result = evaluator.evaluate(pred_tracks, gt_tracks)
    
    print("\n" + "=" * 70)
    print("综合评估结果摘要:")
    print("=" * 70)
    print(f"  MOTA:           {result['MOTA']:.4f}  (越高越好，最高 1.0)")
    print(f"  MOTP:           {result['MOTP']:.4f}  像素 (越低越好)")
    print(f"  IDF1:           {result['IDF1']:.4f}  (越高越好，最高 1.0)")
    print(f"  精准度:         {result['precision']:.4f}")
    print(f"  召回率:         {result['recall']:.4f}")
    print("=" * 70)
    
    # ========== 步骤 7: 解释评估结果 ==========
    print("\n评估指标解释:")
    print("  - MOTA > 0.8:   很好的跟踪性能")
    print("  - MOTA > 0.6:   良好的跟踪性能")
    print("  - MOTA > 0.4:   中等的跟踪性能")
    print("  - MOTA < 0.4:   较差的跟踪性能")
    print("  - IDF1 > 0.8:   很好的 ID 保留性能")
    print("  - IDF1 > 0.6:   良好的 ID 保留性能")


def example_detailed_analysis():
    """
    示例 2: 详细的跟踪分析
    
    本示例展示如何进行更详细的跟踪性能分析。
    """
    print("\n" + "=" * 70)
    print("示例 2: 详细的跟踪性能分析")
    print("=" * 70)
    
    # 初始化评估器
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    
    # 准备数据（更复杂的场景）
    print("\n准备复杂跟踪数据...")
    
    # 预测数据：包含误报和ID切换
    pred_tracks = [
        [{"track_id": 1, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
        [
            {"track_id": 1, "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}},
            {"track_id": 3, "bbox": {"x1": 200, "y1": 200, "x2": 250, "y2": 250}}  # 误报
        ],
        [{"track_id": 2, "bbox": {"x1": 30, "y1": 30, "x2": 70, "y2": 70}}]  # ID 切换
    ]
    
    # 真值数据
    gt_tracks = [
        [{"track_id": 1, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
        [{"track_id": 1, "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}}],
        [{"track_id": 1, "bbox": {"x1": 30, "y1": 30, "x2": 70, "y2": 70}}]
    ]
    
    # 进行评估
    result = evaluator.evaluate(pred_tracks, gt_tracks)
    
    print("\n分析结果:")
    print(f"  检测精准度: {result['precision']:.2%}")
    print(f"  检测召回率: {result['recall']:.2%}")
    print(f"  MOTA 得分:  {result['MOTA']:.4f}")
    print(f"  IDF1 得分:  {result['IDF1']:.4f}")
    print(f"  MOTP 得分:  {result['MOTP']:.4f}")
    
    # 分析可能的问题
    print("\n问题分析:")
    if result['MOTA'] < 0.6:
        print("  ⚠ MOTA 较低，可能存在：")
        print("    - 漏报（遗漏了一些目标）")
        print("    - 误报（错误地检测了目标）")
    
    if result['IDF1'] < 0.7:
        print("  ⚠ IDF1 较低，可能存在：")
        print("    - 频繁的 ID 切换")
        print("    - 跟踪碎片化")
        print("    - ID 分配不准确")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Vision Framework - 跟踪评估示例")
    print("=" * 70)
    print("\n使用标准跟踪评估指标（MOTA、MOTP、IDF1）评估跟踪算法性能")
    print("=" * 70)
    
    try:
        example_basic_evaluation()
        example_detailed_analysis()
        
        print("\n" + "=" * 70)
        print("所有示例运行完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n提示:")
    print("1. 跟踪数据格式必须一致（帧数和 bbox 格式）")
    print("2. IoU 阈值可根据场景调整（通常为 0.5）")
    print("3. 使用真实的检测和跟踪数据可获得更准确的评估结果")


if __name__ == "__main__":
    main()
