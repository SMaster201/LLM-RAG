"""
Qwen RAG System - A document Q&A system using Qwen models and RAG
支援讀取說明書、使用手冊和電子書，並回答相關問題
"""

import os
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import warnings
warnings.filterwarnings('ignore')

class QwenRAGSystem:
    """Qwen模型結合RAG的問答系統"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "auto"):
        """
        初始化Qwen RAG系統
        
        Args:
            model_name: Hugging Face上的Qwen模型名稱
            device: 運行設備 ('cuda', 'cpu', 或 'auto')
        """
        print(f"🚀 正在載入模型: {model_name}")
        
        # 設定設備
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"📱 使用設備: {self.device}")
        
        # 準備本地模型儲存路徑 (./model/<repo_id 替換為 __>)
        root_dir = Path(os.getcwd()) / "model"
        root_dir.mkdir(exist_ok=True)
        safe_name = model_name.replace('/', '__')
        local_model_dir = root_dir / safe_name
        
        # 若本地尚未存在則下載
        if not local_model_dir.exists() or not any(local_model_dir.glob('*.bin')) and not any(local_model_dir.glob('*.safetensors')):
            print(f"🚀 下載模型到 {local_model_dir} ...")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # Windows 避免 symlink 問題
            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
        else:
            print(f"📦 本地模型已存在: {local_model_dir}")
        
        # 載入tokenizer和模型（從本地資料夾）
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(local_model_dir),
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(local_model_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("✅ 模型載入完成")
        
        # 初始化embedding模型
        print("🔧 正在載入Embedding模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 初始化向量存儲
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        print("✅ 系統初始化完成\n")
    
    def load_documents(self, file_paths: List[str]):
        """
        載入文件並建立向量數據庫
        
        Args:
            file_paths: 文件路徑列表（支援PDF和TXT）
        """
        print("📚 開始載入文件...")
        documents = []
        
        for file_path in file_paths:
            print(f"  📄 讀取: {os.path.basename(file_path)}")
            
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"  ⚠️  不支援的文件格式: {file_path}")
                continue
            
            docs = loader.load()
            documents.extend(docs)
        
        print(f"✅ 共載入 {len(documents)} 個文件段落")
        
        # 分割文本
        print("✂️  正在分割文本...")
        splits = self.text_splitter.split_documents(documents)
        print(f"✅ 分割成 {len(splits)} 個文本塊")
        
        # 建立向量數據庫
        print("🗄️  正在建立向量數據庫...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("✅ 向量數據庫建立完成\n")
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        根據查詢檢索相關上下文
        
        Args:
            query: 用戶查詢
            k: 返回最相關的k個文本塊
            
        Returns:
            拼接的上下文文本
        """
        if self.vectorstore is None:
            return ""
        
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def generate_answer(self, query: str, context: str = None) -> str:
        """
        使用Qwen模型生成答案
        
        Args:
            query: 用戶問題
            context: 檢索到的上下文（可選）
            
        Returns:
            生成的答案
        """
        if context:
            prompt = f"""根據以下文件內容回答問題。如果文件中沒有相關資訊，請誠實地說不知道。

文件內容:
{context}

問題: {query}

答案:"""
        else:
            prompt = query
        
        # 使用Qwen的chat模板
        messages = [
            {"role": "system", "content": "你是一個專業的助手，能夠根據提供的文件內容準確回答問題。"},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def ask(self, question: str, use_rag: bool = True) -> str:
        """
        問答接口
        
        Args:
            question: 用戶問題
            use_rag: 是否使用RAG（檢索增強）
            
        Returns:
            答案
        """
        if use_rag and self.vectorstore is not None:
            print(f"\n❓ 問題: {question}")
            print("🔍 正在檢索相關內容...")
            context = self.retrieve_context(question)
            print("🤖 正在生成答案...")
            answer = self.generate_answer(question, context)
        else:
            print(f"\n❓ 問題: {question}")
            print("🤖 正在生成答案（不使用RAG）...")
            answer = self.generate_answer(question)
        
        print(f"💡 答案: {answer}\n")
        return answer


def main():
    """主函數 - 示例用法"""
    
    print("=" * 60)
    print("Qwen RAG 文件問答系統")
    print("=" * 60 + "\n")
    
    # 初始化系統（使用較小的Qwen模型）
    # 可選的7B以下模型:
    # - Qwen/Qwen2.5-1.5B-Instruct
    # - Qwen/Qwen2.5-3B-Instruct
    # - Qwen/Qwen2.5-7B-Instruct
    # - Qwen/Qwen2-1.5B-Instruct
    
    rag_system = QwenRAGSystem(model_name="Qwen/Qwen2.5-3B-Instruct")
    
    print("\n📖 使用說明:")
    print("1. 將您的PDF或TXT文件放在當前目錄")
    print("2. 系統會自動載入並建立知識庫")
    print("3. 您可以開始提問相關問題")
    print("\n" + "=" * 60 + "\n")
    
    # 示例: 載入文件
    # 請將以下路徑替換為您實際的文件路徑
    # file_paths = [
    #     "manual.pdf",      # 使用手冊
    #     "guide.pdf",       # 說明書
    #     "ebook.txt"        # 電子書
    # ]
    # rag_system.load_documents(file_paths)
    
    # 互動式問答
    print("💬 開始互動式問答（輸入 'quit' 或 'exit' 結束）\n")
    
    while True:
        question = input("您的問題: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出', '結束']:
            print("\n👋 再見！")
            break
        
        if not question:
            continue
        
        try:
            answer = rag_system.ask(question)
        except Exception as e:
            print(f"❌ 錯誤: {e}\n")


if __name__ == "__main__":
    main()
