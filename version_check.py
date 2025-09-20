import torch
import subprocess

def check_pytorch_versions():
    print("=== PyTorch Version Information ===")
    
    # PyTorch 버전
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA 사용 가능 여부
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # PyTorch 내 CUDA 버전
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    
    # cuDNN 버전
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    # NVIDIA 드라이버 및 CUDA 버전 (nvidia-smi 명령어 사용)
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print("\n=== NVIDIA-SMI Output (Driver and CUDA Info) ===")
        print(nvidia_smi_output)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        print("nvidia-smi 명령어가 설치되어 있지 않거나 GPU가 지원되지 않을 수 있습니다.")
    
    # 추가: 전체 환경 정보 (torch.utils.collect_env)
    print("\n=== Detailed PyTorch Environment Info ===")
    from torch.utils.collect_env import get_pretty_env_info
    print(get_pretty_env_info())

if __name__ == "__main__":
    check_pytorch_versions()
