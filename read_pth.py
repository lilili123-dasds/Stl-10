import torch
import os

# ===================== 需要自己修改的部分 =====================

pth_file_path = "D:/ljr/Work/project/checkpoints_epoch_100_val/best_val_w/best_val_Adam.pth"  # 这里写你自己的pth文件路径
# ===============================================================

def read_and_analyze_pth(file_path):
    """
    读取并分析 .pth 文件内容
    """
    if not os.path.exists(file_path):
        print(f" 文件不存在: {file_path}")
        return
    
    print(f" 正在读取文件: {file_path}")
    
    try:
        # 加载 .pth 文件 (使用 map_location='cpu' 确保在 CPU 上加载)
        checkpoint = torch.load(file_path, map_location='cpu')
        
        print("\n 文件内容分析:")
        
        # 如果加载的是一个字典，打印它的顶层键
        if isinstance(checkpoint, dict):
            print(f" 字典顶层键 (Keys): {list(checkpoint.keys())}")
            
            # 如果包含 state_dict，打印部分参数名称和形状
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\n state_dict 包含 {len(state_dict)} 个参数:")
                for i, (param_name, param_tensor) in enumerate(state_dict.items()):
                    if i < 5:  # 只打印前5个参数，避免输出太长
                        print(f"  {param_name}: {param_tensor.shape}")
                    elif i == 5:
                        print("  ...")
                if len(state_dict) > 5:
                    print(f"  (共 {len(state_dict)} 个参数，此处省略其余)")
            
            # 如果包含 epoch 或 optimizer 等信息，也打印出来
            for key in ['epoch', 'optimizer', 'lr']:
                if key in checkpoint:
                    print(f"\n {key}: {checkpoint[key]}")
                    
        else:
            # 如果不是字典，直接打印类型和内容
            print(f" 文件类型: {type(checkpoint)}")
            print(f" 内容预览: {checkpoint}")
            
    except Exception as e:
        print(f" 加载文件时出错: {e}")

if __name__ == "__main__":
    read_and_analyze_pth(pth_file_path)