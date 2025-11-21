"""
模型下載輔助腳本
用於預先下載Qwen模型到本地 model/ 資料夾，避免首次使用時等待
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_name: str) -> bool:
    """
    下載並緩存指定的Qwen模型到本地 model/ 資料夾
    
    Args:
        model_name: 模型名稱 (例如: Qwen/Qwen2-0.5B-Instruct)
        
    Returns:
        下載是否成功
    """
    """主函數：支援 CLI 與互動模式

    CLI 範例：
      - 下載全部模型: python download_models.py --all
      - 下載指定模型: python download_models.py --model Qwen/Qwen2-0.5B-Instruct --model Qwen/Qwen2-1.5B-Instruct
      - 保留互動式：直接執行 python download_models.py
    """

    # 可用模型列表 (與 rag_performance_test.py 一致的 7 個模型)
    models = {
        "1": ("Qwen/Qwen2-0.5B-Instruct", "0.5B - 最輕量，速度最快 (~1GB)"),
        "2": ("Qwen/Qwen2-1.5B-Instruct", "1.5B - Qwen2 輕量版 (~3GB)"),
        "3": ("Qwen/Qwen2-7B-Instruct", "7B - Qwen2 大型版 (~14GB)"),
        "4": ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B - Qwen2.5 改進版 (~3GB)"),
        "5": ("Qwen/Qwen2.5-3B-Instruct", "3B - Qwen2.5 中型版 (~6GB)"),
        "6": ("Qwen/Qwen2.5-7B-Instruct", "7B - Qwen2.5 大型版 (~14GB)"),
        "7": ("Qwen/Qwen3-4B-Instruct-2507", "4B - Qwen3 最新版 (~8GB)"),
    }

    parser = argparse.ArgumentParser(description="Download Qwen models to local model/ folder or run interactively")
    parser.add_argument('--all', action='store_true', help='Download all predefined models')
    parser.add_argument('--model', action='append', help='Specify model repo_id to download (can repeat). e.g. --model Qwen/Qwen2-0.5B-Instruct')
    args = parser.parse_args()

    # Non-interactive: --all or --model provided
    if args.all or args.model:
        targets = []
        if args.all:
            targets = [m[0] for m in models.values()]
        if args.model:
            for m in args.model:
                # allow numeric index or repo id
                if m.isdigit() and m in models:
                    targets.append(models[m][0])
                else:
                    targets.append(m)

        success_count = 0
        failed = []
        for repo in targets:
            print('\n' + '=' * 70)
            print(f"📥 下載: {repo}")
            print('=' * 70 + '\n')
            if download_model(repo):
                success_count += 1
            else:
                failed.append(repo)

        print('\n' + '=' * 70)
        print('下載完成')
        print(f'✅ 成功: {success_count}/{len(targets)}')
        if failed:
            print(f'❌ 失敗: {len(failed)}')
            for r in failed:
                print(f'   - {r}')
        print('=' * 70 + '\n')
        return

    # Fallback to original interactive mode
    print("\n" + "=" * 70)
    print("Qwen 模型下載工具 - 下載到本地 model/ 資料夾")
    print("=" * 70 + "\n")
    
    print("請選擇要下載的模型:\n")
    for key, (name, desc) in models.items():
        print(f"{key}. {desc}")
        print(f"   {name}\n")
    
    print("0. 下載所有模型（需要大量時間和空間，約 50GB）\n")
    
    choice = input("請選擇 (0-7): ").strip()
    
    if choice == "0":
        # 下載所有模型
        confirm = input("\n⚠️  這將下載所有 7 個模型（約 50GB），確定嗎？(y/n): ").strip().lower()
        if confirm == 'y':
            print("\n開始下載所有模型...\n")
            success_count = 0
            failed_models = []
            
            for i, (model_name, desc) in models.items():
                print(f"\n{'='*70}")
                print(f"下載進度: {i}/{len(models)}")
                print(f"{'='*70}\n")
                if download_model(model_name):
                    success_count += 1
                else:
                    failed_models.append(f"{model_name} ({desc})")
            
            # 顯示總結
            print("\n" + "=" * 70)
            print("下載總結")
            print("=" * 70)
            print(f"✅ 成功: {success_count}/{len(models)}")
            if failed_models:
                print(f"❌ 失敗: {len(failed_models)}")
                for model in failed_models:
                    print(f"   - {model}")
            print("=" * 70 + "\n")
        else:
            print("已取消")
    
    elif choice in models:
        # 下載單個模型
        model_name, desc = models[choice]
        print(f"\n您選擇了: {desc}")
        print(f"模型: {model_name}")
        confirm = input("\n確定下載嗎？(y/n): ").strip().lower()
        
        if confirm == 'y':
            success = download_model(model_name)
            if success:
                print("\n✅ 完成！現在可以使用 qwen_rag_system.py 或 rag_performance_test.py 了")
                print(f"\n使用範例:")
                print(f'python qwen_rag_system.py  # 會使用本地已下載的模型')
        else:
            print("已取消")
    
    else:
        print("❌ 無效的選擇，請輸入 0-7")
    
    try:
        # 準備本地模型儲存路徑
        root_dir = Path(os.getcwd()) / "model"
        root_dir.mkdir(exist_ok=True)
        safe_name = model_name.replace('/', '__')
        local_model_dir = root_dir / safe_name
        
        # 檢查是否已存在
        if local_model_dir.exists() and (
            any(local_model_dir.glob('*.bin')) or 
            any(local_model_dir.glob('*.safetensors'))
        ):
            print(f"✅ 模型已存在: {local_model_dir}")
            print("跳過下載\n")
            return True
        
        print(f"📥 下載位置: {local_model_dir}")
        print("📥 正在下載模型（這可能需要幾分鐘到幾十分鐘）...\n")
        
        # 設定環境變數避免 Windows symlink 問題
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        
        # 下載完整模型
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            repo_type="model"
        )
        
        print("\n✅ 模型下載完成")
        print(f"✅ 儲存位置: {local_model_dir}\n")
        
        print("=" * 70)
        print("下載完成！模型已緩存到本地 model/ 資料夾")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下載失敗: {e}")
        print("\n可能的原因:")
        print("1. 網路連接問題")
        print("2. Hugging Face訪問受限")
        print("3. 硬碟空間不足")
        print("4. 權限不足（Windows 可能需要管理員權限）")
        return False


def main():
    """主函數"""
    
    print("\n" + "=" * 70)
    print("Qwen 模型下載工具 - 下載到本地 model/ 資料夾")
    print("=" * 70 + "\n")
    
    # 可用模型列表 (與 rag_performance_test.py 一致的 7 個模型)
    models = {
        "1": ("Qwen/Qwen2-0.5B-Instruct", "0.5B - 最輕量，速度最快 (~1GB)"),
        "2": ("Qwen/Qwen2-1.5B-Instruct", "1.5B - Qwen2 輕量版 (~3GB)"),
        "3": ("Qwen/Qwen2-7B-Instruct", "7B - Qwen2 大型版 (~14GB)"),
        "4": ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B - Qwen2.5 改進版 (~3GB)"),
        "5": ("Qwen/Qwen2.5-3B-Instruct", "3B - Qwen2.5 中型版 (~6GB)"),
        "6": ("Qwen/Qwen2.5-7B-Instruct", "7B - Qwen2.5 大型版 (~14GB)"),
        "7": ("Qwen/Qwen3-4B-Instruct-2507", "4B - Qwen3 最新版 (~8GB)"),
    }
    
    print("請選擇要下載的模型:\n")
    for key, (name, desc) in models.items():
        print(f"{key}. {desc}")
        print(f"   {name}\n")
    
    print("0. 下載所有模型（需要大量時間和空間，約 50GB）\n")
    
    choice = input("請選擇 (0-7): ").strip()
    
    if choice == "0":
        # 下載所有模型
        confirm = input("\n⚠️  這將下載所有 7 個模型（約 50GB），確定嗎？(y/n): ").strip().lower()
        if confirm == 'y':
            print("\n開始下載所有模型...\n")
            success_count = 0
            failed_models = []
            
            for i, (model_name, desc) in models.items():
                print(f"\n{'='*70}")
                print(f"下載進度: {i}/{len(models)}")
                print(f"{'='*70}\n")
                if download_model(model_name):
                    success_count += 1
                else:
                    failed_models.append(f"{model_name} ({desc})")
            
            # 顯示總結
            print("\n" + "=" * 70)
            print("下載總結")
            print("=" * 70)
            print(f"✅ 成功: {success_count}/{len(models)}")
            if failed_models:
                print(f"❌ 失敗: {len(failed_models)}")
                for model in failed_models:
                    print(f"   - {model}")
            print("=" * 70 + "\n")
        else:
            print("已取消")
    
    elif choice in models:
        # 下載單個模型
        model_name, desc = models[choice]
        print(f"\n您選擇了: {desc}")
        print(f"模型: {model_name}")
        confirm = input("\n確定下載嗎？(y/n): ").strip().lower()
        
        if confirm == 'y':
            success = download_model(model_name)
            if success:
                print("\n✅ 完成！現在可以使用 qwen_rag_system.py 或 rag_performance_test.py 了")
                print(f"\n使用範例:")
                print(f'python qwen_rag_system.py  # 會使用本地已下載的模型')
        else:
            print("已取消")
    
    else:
        print("❌ 無效的選擇，請輸入 0-7")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  下載已中斷")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
