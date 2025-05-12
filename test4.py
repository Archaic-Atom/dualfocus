import torch
import os


def compare_saved_weights(file_path1: str, file_path2: str, atol: float = 1e-6, rtol: float = 1e-4):
    """
    比较两个保存的模型权重（state_dict）文件，并指出哪些参数不同。

    Args:
        file_path1 (str): 第一个权重文件的路径。
        file_path2 (str): 第二个权重文件的路径。
        atol (float): torch.allclose 使用的绝对容忍度。
        rtol (float): torch.allclose 使用的相对容忍度。

    Returns:
        bool: 如果找到差异则返回 True，否则返回 False。
    """
    print(f"\n--- Comparing weights from '{os.path.basename(file_path1)}' and '{os.path.basename(file_path2)}' ---")

    if not os.path.exists(file_path1):
        print(f"Error: File not found - {file_path1}")
        return True  # Indicate error by returning True (differences exist because one is missing)
    if not os.path.exists(file_path2):
        print(f"Error: File not found - {file_path2}")
        return True  # Indicate error

    # 加载权重字典，确保在CPU上加载以避免GPU问题
    try:
        print(f"Loading weights from: {file_path1}")
        weights1 = torch.load(file_path1, map_location='cpu')
        print(f"Successfully loaded {len(weights1)} parameters from {file_path1}")
    except Exception as e:
        print(f"Error loading weights from {file_path1}: {e}")
        return True

    try:
        print(f"Loading weights from: {file_path2}")
        weights2 = torch.load(file_path2, map_location='cpu')
        print(f"Successfully loaded {len(weights2)} parameters from {file_path2}")
    except Exception as e:
        print(f"Error loading weights from {file_path2}: {e}")
        return True

    # 获取所有参数的键名集合，以便处理一个文件有另一个没有的情况
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    all_keys = sorted(list(keys1.union(keys2)))  # 合并并排序所有键名

    differences_found = False

    for key in all_keys:
        param1_exists = key in weights1
        param2_exists = key in weights2

        if param1_exists and param2_exists:
            param1 = weights1[key]
            param2 = weights2[key]

            # 检查形状是否一致
            if param1.shape != param2.shape:
                print(f"Parameter '{key}': Shape mismatch!")
                print(f"  Shape in '{os.path.basename(file_path1)}': {param1.shape}")
                print(f"  Shape in '{os.path.basename(file_path2)}': {param2.shape}")
                differences_found = True
                continue  # 跳过值的比较

            # 比较张量值
            # 使用 allclose 进行浮点数比较，因为直接等号比较可能因精度问题失败
            if not torch.allclose(param1, param2, atol=atol, rtol=rtol):
                print(f"Parameter '{key}': Values differ.")
                differences_found = True
                # (可选) 打印一些差异统计信息
                try:
                    # 确保转换为浮点数进行数值计算，以防是 bfloat16 等类型
                    diff_norm = torch.linalg.norm(param1.float() - param2.float()).item()
                    max_abs_diff = torch.max(torch.abs(param1.float() - param2.float())).item()
                    print(f"  L2 norm of difference: {diff_norm:.4e}")
                    print(f"  Max absolute difference: {max_abs_diff:.4e}")
                    # print(f"  Sample from '{os.path.basename(file_path1)}': {param1.flatten()[:min(3, param1.numel())]}")
                    # print(f"  Sample from '{os.path.basename(file_path2)}': {param2.flatten()[:min(3, param2.numel())]}")
                except Exception as e:
                    print(f"  Could not compute difference stats for '{key}': {e}")
            # else:
            #     print(f"Parameter '{key}': Identical (within tolerance).") # 可选的详细输出

        elif param1_exists:
            print(f"Parameter '{key}': Only exists in '{os.path.basename(file_path1)}'.")
            differences_found = True
        elif param2_exists:
            print(f"Parameter '{key}': Only exists in '{os.path.basename(file_path2)}'.")
            differences_found = True
        else:
            # This case should not happen if all_keys is derived correctly
            print(f"Logic error: Key '{key}' not found in either dictionary.")

    if not differences_found:
        print("\n--- No differences found between the weight files (within tolerance). ---")
    else:
        print("\n--- Differences were found between the weight files. ---")

    return differences_found


# --- 在主脚本中如何使用这个新功能 ---
if __name__ == "__main__":
    # ... (你已有的主脚本代码，包括模型初始化等) ...
    # ... (假设 siglip_extractor 已经初始化并可能已保存了一些权重) ...

    # --- 比较两个权重文件 ---
    # 示例路径，你需要替换成你实际的权重文件路径
    # 假设你已经通过 siglip_extractor.save_trainable_weight() 保存了两个不同的权重文件
    # 例如，在不同训练阶段保存的权重，或者修改了部分权重后再保存的

    # 为了演示，我们先创建两个略有不同的示例权重文件
    # 在实际使用中，这两个文件会是你训练过程中保存的检查点

    # 模拟第一个权重文件 (可以是你之前保存的)
    save_file_path1 = '/data1/sunmingyu/xiaoxia_laboratory/my_study/dualfocus/training_output/dualfocus_msrvtt_exp01/checkpoints/qvit_step_40.pth'
    # save_file_path1 = '/data1/sunmingyu/xiaoxia_laboratory/my_study/dualfocus/test/qvit_trainable_weight.pt'

    # 模拟第二个权重文件 (可以是你之后保存的，或者手动修改过的)
    # 这里我们假设 siglip_extractor 已经加载了 save_file_path1 的权重
    # 然后我们稍微修改一下它的可训练权重，再保存到 save_file_path2
    # save_file_path2 = "./test/vit_trainable_weights_modified.pth"
    save_file_path2 = '/data1/sunmingyu/xiaoxia_laboratory/my_study/dualfocus/training_output/dualfocus_msrvtt_exp01/checkpoints/qvit_step_20.pth'

    # 确保 siglip_extractor 存在并且 EMBEDDING_TYPE 设置为 'num' 以便有可训练的 temporal_embedding
    # 这部分是为了创建示例文件，如果你的文件已存在，则不需要
    # if 'siglip_extractor' in locals() and hasattr(siglip_extractor, 'pos_encoder') and \
    #    hasattr(siglip_extractor.pos_encoder, 'temporal_embedding') and \
    #    isinstance(siglip_extractor.pos_encoder.temporal_embedding, torch.nn.Embedding):

    #     print(f"\nCreating example weight files for comparison...")
    #     # 1. 保存原始权重到 path1 (如果它还不存在或你想覆盖)
    #     if not os.path.exists(save_file_path1): # 或者总是保存以确保最新
    #          siglip_extractor.save_trainable_weight(save_file_path1)
    #          print(f"Saved initial weights to {save_file_path1}")

    #     # 2. 修改权重并保存到 path2
    #     original_weights_for_comparison = {}
    #     modified_weights_for_comparison = {}

    #     with torch.no_grad():
    #         # 复制一份原始权重，以便比较后恢复
    #         # for name, param in siglip_extractor.named_parameters():
    #         #     if param.requires_grad:
    #         #         original_weights_for_comparison[name] = param.data.clone()

    #         # 假设 temporal_embedding 是可训练的
    #         # 注意：siglip_extractor.save_trainable_weight() 内部会过滤requires_grad=True的参数
    #         # 所以我们只需要修改模型中实际可训练的参数
    #         temp_emb_weight = siglip_extractor.pos_encoder.temporal_embedding.weight
    #         value_to_add = 0.1
    #         temp_emb_weight.data += value_to_add # 对权重进行微小修改
    #         print(f"Modified 'pos_encoder.temporal_embedding.weight' by adding {value_to_add}.")

    #         # 保存修改后的权重
    #         # siglip_extractor.save_trainable_weight(save_file_path2)
    #         # print(f"Saved modified weights to {save_file_path2}")

    #         # 恢复原始权重到模型，以免影响后续操作
    #         # for name, original_param_data in original_weights_for_comparison.items():
    #         #     # 需要找到模型中对应的参数并恢复
    #         #     # 这部分有点复杂，因为 name_parameters() 返回的 name 可能与 state_dict key 不完全一致
    #         #     # 对于简单情况，如 pos_encoder.temporal_embedding.weight，可以直接恢复
    #         #     if name == "pos_encoder.temporal_embedding.weight": # 假设这是唯一可训练的
    #         #         siglip_extractor.pos_encoder.temporal_embedding.weight.data.copy_(original_param_data)
    #         #         print("Restored original weight to 'pos_encoder.temporal_embedding.weight' in memory.")

    # else:
    #     print("\nSkipping creation of example weight files: siglip_extractor or trainable embedding not found/configured as expected.")
    #     print("Please ensure save_file_path1 and save_file_path2 point to valid weight files for comparison.")

    # 现在比较这两个文件
    if os.path.exists(save_file_path1) and os.path.exists(save_file_path2):
        differences = compare_saved_weights(save_file_path1, save_file_path2)
        if differences:
            print("\nComparison complete: Differences were detected.")
        else:
            print("\nComparison complete: No differences detected (or files were identical).")
    else:
        print(f"\nCannot compare: One or both weight files do not exist.")
        print(f"  '{save_file_path1}' exists: {os.path.exists(save_file_path1)}")
        print(f"  '{save_file_path2}' exists: {os.path.exists(save_file_path2)}")
        print("Please ensure the paths are correct and the files have been generated.")

    print("\n--- Example Finished ---")