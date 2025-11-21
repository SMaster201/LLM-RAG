"""
Qwen RAG 完整效能測試
測試模型讀取 PDF、回答問題，並記錄所有 VRAM 和效能指標
"""

import os
import sys
import time
import json
import torch
import gc
import psutil
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

# 嘗試導入 VRAM 監控工具
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False
    print("⚠️ NVIDIA GPU 不可用或 pynvml 未安裝，將只記錄 CPU 記憶體")

from qwen_rag_system import QwenRAGSystem


class PerformanceMonitor:
    """效能監控器"""
    
    def __init__(self):
        self.has_gpu = NVML_AVAILABLE and torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline_vram = 0
        self.peak_vram = 0
        self.process = psutil.Process()
        
    def get_vram_mb(self) -> float:
        """獲取當前 VRAM 使用量 (MB)"""
        if self.has_gpu:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                return info.used / 1024 / 1024
            except:
                return 0
        else:
            # 返回 CPU RAM 使用量
            return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_memory_mb(self) -> float:
        """獲取 CPU 記憶體使用量 (MB)"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def set_baseline(self):
        """設定基線 VRAM"""
        self.baseline_vram = self.get_vram_mb()
        self.peak_vram = self.baseline_vram
        
    def update_peak(self):
        """更新峰值 VRAM"""
        current = self.get_vram_mb()
        if current > self.peak_vram:
            self.peak_vram = current
            
    def get_stats(self) -> Dict[str, float]:
        """獲取統計數據"""
        current = self.get_vram_mb()
        return {
            'baseline_vram_mb': round(self.baseline_vram, 2),
            'current_vram_mb': round(current, 2),
            'peak_vram_mb': round(self.peak_vram, 2),
            'vram_growth_mb': round(self.peak_vram - self.baseline_vram, 2),
            'cpu_memory_mb': round(self.get_cpu_memory_mb(), 2)
        }


def clear_memory():
    """清理記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


def test_rag_with_pdf(model_name: str, pdf_folder: str, questions: List[str]) -> Dict[str, Any]:
    """
    測試 RAG 系統：載入 PDF、回答問題
    
    Args:
        model_name: 模型名稱
        pdf_folder: PDF 資料夾路徑
        questions: 要問的問題列表
        
    Returns:
        包含所有測試結果的字典
    """
    print("=" * 80)
    print(f"測試模型: {model_name}")
    print(f"PDF 資料夾: {pdf_folder}")
    print(f"問題數量: {len(questions)}")
    print("=" * 80 + "\n")
    
    monitor = PerformanceMonitor()
    results = {
        'model_name': model_name,
        'pdf_folder': pdf_folder,
        'questions': questions,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'timestamp': datetime.now().isoformat()
    }
    
    # 清理記憶體
    clear_memory()
    monitor.set_baseline()
    
    # 1. 載入模型
    print("📥 階段 1: 載入模型")
    model_load_start = time.time()
    
    try:
        rag_system = QwenRAGSystem(model_name=model_name, device="auto")
        model_load_time = time.time() - model_load_start
        monitor.update_peak()
        
        results['model_load_time_sec'] = round(model_load_time, 2)
        results['model_load_success'] = True
        print(f"✅ 模型載入完成，耗時: {model_load_time:.2f} 秒")
        
        # 記錄模型載入後的 VRAM
        model_loaded_stats = monitor.get_stats()
        results['after_model_load'] = model_loaded_stats
        print(f"   VRAM: {model_loaded_stats['current_vram_mb']} MB")
        print()
        
    except Exception as e:
        results['model_load_success'] = False
        results['model_load_error'] = str(e)
        print(f"❌ 模型載入失敗: {e}")
        return results
    
    # 2. 載入 PDF 文件
    print("📚 階段 2: 載入 PDF 文件")
    pdf_load_start = time.time()
    
    try:
        # 掃描 PDF 資料夾
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"在 {pdf_folder} 中找不到 PDF 文件")
        
        print(f"   找到 {len(pdf_files)} 個 PDF 文件")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        
        # 載入文件到 RAG 系統
        rag_system.load_documents([str(f) for f in pdf_files])
        pdf_load_time = time.time() - pdf_load_start
        monitor.update_peak()
        
        results['pdf_count'] = len(pdf_files)
        results['pdf_files'] = [f.name for f in pdf_files]
        results['pdf_load_time_sec'] = round(pdf_load_time, 2)
        results['pdf_load_success'] = True
        print(f"✅ PDF 載入完成，耗時: {pdf_load_time:.2f} 秒")
        
        # 記錄 PDF 載入後的 VRAM
        pdf_loaded_stats = monitor.get_stats()
        results['after_pdf_load'] = pdf_loaded_stats
        print(f"   VRAM: {pdf_loaded_stats['current_vram_mb']} MB")
        print(f"   RAG 額外 VRAM: {pdf_loaded_stats['current_vram_mb'] - model_loaded_stats['current_vram_mb']:.2f} MB")
        print()
        
    except Exception as e:
        results['pdf_load_success'] = False
        results['pdf_load_error'] = str(e)
        print(f"❌ PDF 載入失敗: {e}")
        return results
    
    # 3. 問答測試
    print("❓ 階段 3: 問答測試")
    qa_start = time.time()
    
    qa_results = []
    total_retrieval_time = 0
    total_generation_time = 0
    total_tokens = 0
    
    try:
        for i, question in enumerate(questions, 1):
            print(f"\n   問題 {i}/{len(questions)}: {question}")
            
            # 先檢索相關內容
            retrieval_start = time.time()
            context = rag_system.retrieve_context(question, k=3)
            retrieval_time = time.time() - retrieval_start
            monitor.update_peak()
            
            # 生成答案
            generation_start = time.time()
            answer = rag_system.generate_answer(question, context)
            generation_time = time.time() - generation_start
            monitor.update_peak()
            
            # 計算 tokens（粗略估計）
            answer_tokens = len(answer.split())
            tokens_per_sec = answer_tokens / generation_time if generation_time > 0 else 0
            
            total_retrieval_time += retrieval_time
            total_generation_time += generation_time
            total_tokens += answer_tokens
            
            qa_result = {
                'question': question,
                'answer': answer,
                'retrieval_time_sec': round(retrieval_time, 2),
                'generation_time_sec': round(generation_time, 2),
                'answer_length': len(answer),
                'answer_tokens_estimate': answer_tokens,
                'tokens_per_sec': round(tokens_per_sec, 2)
            }
            qa_results.append(qa_result)
            
            print(f"   ✓ 檢索: {retrieval_time:.2f}秒 | 生成: {generation_time:.2f}秒 | 速度: {tokens_per_sec:.2f} t/s")
            print(f"   📝 答案: {answer}")
        
        qa_total_time = time.time() - qa_start
        avg_tokens_per_sec = total_tokens / total_generation_time if total_generation_time > 0 else 0
        
        results['qa_results'] = qa_results
        results['total_retrieval_time_sec'] = round(total_retrieval_time, 2)
        results['total_generation_time_sec'] = round(total_generation_time, 2)
        results['qa_total_time_sec'] = round(qa_total_time, 2)
        results['avg_tokens_per_sec'] = round(avg_tokens_per_sec, 2)
        results['total_questions'] = len(questions)
        results['qa_success'] = True
        
        print(f"\n✅ 所有問答完成，總耗時: {qa_total_time:.2f} 秒")
        print(f"   平均生成速度: {avg_tokens_per_sec:.2f} tokens/秒")
        
        # 記錄問答後的 VRAM
        qa_stats = monitor.get_stats()
        results['after_qa'] = qa_stats
        print(f"   峰值 VRAM: {qa_stats['peak_vram_mb']} MB")
        print()
        
    except Exception as e:
        results['qa_success'] = False
        results['qa_error'] = str(e)
        print(f"❌ 問答失敗: {e}")
        return results
    
    # 4. 最終統計
    final_stats = monitor.get_stats()
    results['final_stats'] = final_stats
    results['total_vram_growth_mb'] = final_stats['vram_growth_mb']
    
    # RAG 成本評估
    if 'after_model_load' in results and 'after_pdf_load' in results:
        rag_overhead = results['after_pdf_load']['current_vram_mb'] - results['after_model_load']['current_vram_mb']
        results['rag_overhead_mb'] = round(rag_overhead, 2)
    
    print("=" * 80)
    print("📊 測試完成")
    print("=" * 80)
    
    return results


def generate_excel_report(results_list: List[Dict[str, Any]], output_file: str):
    """
    生成 Excel 報告
    
    Args:
        results_list: 測試結果列表
        output_file: 輸出 Excel 檔案路徑
    """
    print(f"\n📊 生成 Excel 報告: {output_file}")
    
    # 準備主要數據表
    main_data = []
    for r in results_list:
        if not r.get('qa_success'):
            continue
            
        row = {
            '模型版本': r['model_name'],
            '執行設備': r['device'],
            '執行任務類型': 'RAG 問答',
            'PDF 數量': r.get('pdf_count', 0),
            '問題數量': r.get('total_questions', 0),
            '模型載入時間 (秒)': r.get('model_load_time_sec', 0),
            'PDF 載入時間 (秒)': r.get('pdf_load_time_sec', 0),
            '總檢索時間 (秒)': r.get('total_retrieval_time_sec', 0),
            '總生成時間 (秒)': r.get('total_generation_time_sec', 0),
            '總耗時 (秒)': r.get('qa_total_time_sec', 0),
            '靜置 VRAM (MB)': r['final_stats']['baseline_vram_mb'],
            '峰值 VRAM (MB)': r['final_stats']['peak_vram_mb'],
            'VRAM 增長量 (MB)': r['final_stats']['vram_growth_mb'],
            'RAG 額外 VRAM (MB)': r.get('rag_overhead_mb', 0),
            'CPU 記憶體 (MB)': r['final_stats']['cpu_memory_mb'],
            '平均生成速度 (tokens/秒)': r.get('avg_tokens_per_sec', 0),
            '測試時間': r['timestamp']
        }
        main_data.append(row)
    
    # 創建 Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 主要數據表
        if main_data:
            df_main = pd.DataFrame(main_data)
            df_main.to_excel(writer, sheet_name='效能測試結果', index=False)
        
        # VRAM 階段分析
        vram_data = []
        for r in results_list:
            if not r.get('qa_success'):
                continue
            
            stages = [
                ('基線', r['final_stats']['baseline_vram_mb']),
                ('模型載入後', r.get('after_model_load', {}).get('current_vram_mb', 0)),
                ('PDF 載入後', r.get('after_pdf_load', {}).get('current_vram_mb', 0)),
                ('問答後峰值', r['final_stats']['peak_vram_mb'])
            ]
            
            for stage_name, vram in stages:
                vram_data.append({
                    '模型': r['model_name'],
                    '階段': stage_name,
                    'VRAM (MB)': vram
                })
        
        if vram_data:
            df_vram = pd.DataFrame(vram_data)
            df_vram.to_excel(writer, sheet_name='VRAM 階段分析', index=False)
        
        # RAG 成本評估
        rag_cost_data = []
        for r in results_list:
            if not r.get('qa_success'):
                continue
            
            rag_cost_data.append({
                '模型': r['model_name'],
                '基礎模型 VRAM (MB)': r.get('after_model_load', {}).get('current_vram_mb', 0),
                '加入 RAG 後 VRAM (MB)': r.get('after_pdf_load', {}).get('current_vram_mb', 0),
                'RAG 額外成本 (MB)': r.get('rag_overhead_mb', 0),
                'RAG 成本比例 (%)': round(r.get('rag_overhead_mb', 0) / r.get('after_model_load', {}).get('current_vram_mb', 1) * 100, 2) if r.get('after_model_load', {}).get('current_vram_mb', 0) > 0 else 0,
                'PDF 數量': r.get('pdf_count', 0),
                'PDF 載入時間 (秒)': r.get('pdf_load_time_sec', 0)
            })
        
        if rag_cost_data:
            df_rag = pd.DataFrame(rag_cost_data)
            df_rag.to_excel(writer, sheet_name='RAG 成本評估', index=False)
        
        # 速度分析
        speed_data = []
        for r in results_list:
            if not r.get('qa_success'):
                continue
            
            speed_data.append({
                '模型': r['model_name'],
                '執行設備': r['device'],
                '問題數量': r.get('total_questions', 0),
                '總檢索時間 (秒)': r.get('total_retrieval_time_sec', 0),
                '總生成時間 (秒)': r.get('total_generation_time_sec', 0),
                '總響應時間 (秒)': r.get('qa_total_time_sec', 0),
                '平均生成速度 (tokens/秒)': r.get('avg_tokens_per_sec', 0),
                '是否滿足即時需求': '是' if r.get('qa_total_time_sec', 999) < 10 else '否',
                '速度評級': '快' if r.get('avg_tokens_per_sec', 0) > 20 else '中' if r.get('avg_tokens_per_sec', 0) > 10 else '慢'
            })
        
        if speed_data:
            df_speed = pd.DataFrame(speed_data)
            df_speed.to_excel(writer, sheet_name='速度分析', index=False)
        
        # 問題與答案
        qa_data = []
        for r in results_list:
            if not r.get('qa_success'):
                continue
            
            # 為每個問題創建一行
            for qa in r.get('qa_results', []):
                qa_data.append({
                    '模型': r['model_name'],
                    '問題': qa['question'],
                    '答案': qa['answer'],
                    '檢索時間 (秒)': qa['retrieval_time_sec'],
                    '生成時間 (秒)': qa['generation_time_sec'],
                    '生成速度 (tokens/秒)': qa['tokens_per_sec'],
                    '答案長度 (字元)': qa['answer_length'],
                    'PDF 文件': ', '.join(r.get('pdf_files', []))
                })
        
        if qa_data:
            df_qa = pd.DataFrame(qa_data)
            df_qa.to_excel(writer, sheet_name='問題與答案', index=False)
    
    print(f"✅ Excel 報告已保存: {output_file}")


def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("Qwen RAG 完整效能測試")
    print("=" * 80 + "\n")
    
    # 配置
    PDF_FOLDER = "PDF"  # PDF 資料夾路徑

    QUESTIONS = None
    try:
        import json
        qpath = Path(os.getcwd()) / "test_questions.json"
        if qpath.exists():
            with open(qpath, 'r', encoding='utf-8') as f:
                items = json.load(f)
                # items 預期為 list of {"question": ..., ...}
                QUESTIONS = [it.get('question', '').strip() for it in items if it.get('question')]
    except Exception:
        QUESTIONS = None

    if not QUESTIONS:
        QUESTIONS = [
            "報告中提到全球正從「全球化」轉向「再全球化」（Re-globalization），且面臨「川普 2.0」帶來的關稅與供應鏈重組壓力。請分析這種國際局勢如何具體影響台灣在「半導體」與「工具機」這兩個關鍵產業的技術布局策略？政府又提出了哪些具體的「供應鏈韌性」或「自主化」措施來應對這些外部衝擊？",
            "白皮書強調「數位轉型」與「淨零轉型」是台灣產業的雙軸核心。請詳細說明在「材化領域」或「智慧製造」中，如何具體利用「AI 技術」（如生成式 AI、機器學習）來同時達成「製程效率提升」與「節能減碳」這兩個看似衝突的目標？請舉出報告中提到的具體技術案例（例如化工製程或金屬加工）佐證。",
            "針對「五大信賴產業」中的次世代通訊，報告提出了「地面」與「非地面（NTN）」網路的整合願景。請深入解釋台灣在「低軌衛星（LEO）」地面設備的關鍵技術缺口為何（如射頻晶片、相控陣列天線）？以及「軟體定義無線電（SDR）」技術如何在建構這種「3D 立體通訊網路」中扮演核心角色？",
            "在摩爾定律逼近極限的背景下，白皮書指出「異質整合封裝」與「矽光子（CPO）」是未來的關鍵。請分析台灣在發展這些技術時，面臨了哪些「設備」與「材料」上的自主化挑戰（例如散熱基板材料、檢測設備）？政府的「晶創台灣方案」與相關科專計畫又是如何協助廠商突破這些被國外大廠壟斷的瓶頸？",
            "針對「健康台灣」的願景，白皮書中提到的「新藥開發」與「醫療器材」如何擺脫傳統研發模式？請具體說明「AI 運算」與「生醫晶片」技術如何被應用於縮短新藥開發週期（如 mRNA 藥物），以及實現「非侵入式」或「居家化」的精準醫療（如眼科滴劑或高齡照護）？"
        ]
    OUTPUT_JSON = "rag_performance_test_results.json"
    OUTPUT_EXCEL = "RAG效能測試報告.xlsx"
    
    # 測試模型列表（已下載的所有模型）
    MODELS = [
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen3-4B-Instruct-2507",
        # 7B 模型需要更多 VRAM，如果顯卡足夠可取消註解：
         "Qwen/Qwen2-7B-Instruct",
         "Qwen/Qwen2.5-7B-Instruct",
    ]
    
    # 檢查 PDF 資料夾
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ 錯誤: PDF 資料夾不存在: {PDF_FOLDER}")
        print(f"請創建資料夾並放入 PDF 文件")
        return
    
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ 錯誤: 在 {PDF_FOLDER} 中找不到 PDF 文件")
        return
    
    print(f"✅ 找到 {len(pdf_files)} 個 PDF 文件:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()
    
    # 執行測試
    all_results = []
    
    for i, model_name in enumerate(MODELS, 1):
        print(f"\n{'='*80}")
        print(f"測試進度: {i}/{len(MODELS)}")
        print(f"{'='*80}\n")
        
        try:
            result = test_rag_with_pdf(model_name, PDF_FOLDER, QUESTIONS)
            all_results.append(result)
            
            # 保存中間結果
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"\n❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理記憶體
        if i < len(MODELS):
            print("\n🧹 清理記憶體...")
            clear_memory()
            time.sleep(3)
    
    # 生成報告
    print("\n" + "=" * 80)
    print("生成報告")
    print("=" * 80)
    
    # JSON 報告
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 報告已保存: {OUTPUT_JSON}")
    
    # Excel 報告
    generate_excel_report(all_results, OUTPUT_EXCEL)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("測試摘要")
    print("=" * 80 + "\n")
    
    for result in all_results:
        if result.get('qa_success'):
            print(f"✅ {result['model_name']}")
            print(f"   設備: {result['device']}")
            print(f"   問題數量: {result.get('total_questions', 0)}")
            print(f"   總耗時: {result.get('qa_total_time_sec', 0):.2f} 秒")
            print(f"   平均生成速度: {result.get('avg_tokens_per_sec', 0):.2f} tokens/秒")
            print(f"   峰值 VRAM: {result['final_stats']['peak_vram_mb']} MB")
            print(f"   RAG 額外成本: {result.get('rag_overhead_mb', 0)} MB")
            print()
        else:
            print(f"❌ {result['model_name']} - 測試失敗")
            print()
    
    print("=" * 80)
    print("所有測試完成！")
    print(f"詳細報告: {OUTPUT_EXCEL}")
    print("=" * 80)


if __name__ == "__main__":
    main()
