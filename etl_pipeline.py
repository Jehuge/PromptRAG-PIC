"""
ETL Pipeline：数据清洗与结构化模块
从 Excel/CSV 读取原始提示词，通过 Qwen 3 解析成结构化 JSON
"""
import pandas as pd
import json
import jsonlines
import os
from typing import List, Dict, Optional
from tqdm import tqdm
from ollama_client import OllamaClient
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


class ETLPipeline:
    """ETL 数据处理管道"""
    
    def __init__(self, ollama_client: OllamaClient = None):
        self.client = ollama_client or OllamaClient()
        self.system_prompt = self._get_system_prompt()
        
        # 确保目录存在
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    def _get_system_prompt(self) -> str:
        """获取用于结构化提取的系统提示词"""
        return """你是一位资深的美术总监和提示词工程师。你的任务是将用户提供的原始绘图提示词（可能是中文或英文）解析成结构化的 JSON 格式，并转换为中文。

输出要求：
1. 必须返回纯 JSON 格式，不要包含任何解释性文字、markdown 代码块标记或其他内容
2. JSON 必须包含以下字段：
   - subject: 画面的核心主体内容（字符串，必须用中文）
   - art_style: 艺术风格（字符串，必须用中文，如 "赛博朋克", "油画", "印象派"）
   - visual_elements: 视觉元素列表（数组，必须用中文，如 ["霓虹灯", "雨", "猫咪", "城市街道"]）
   - mood: 氛围/情绪（字符串，必须用中文，如 "阴郁", "充满活力", "神秘"）
   - technical: 技术参数列表（数组，必须用中文，如 ["8k", "杰作", "光线追踪", "高度细节"]）
   - raw: 原始提示词文本（字符串，必须转换为中文版本）

重要：
- 无论原始提示词是中文还是英文，所有字段都必须转换为中文
- 如果原始提示词是英文，请准确理解其含义并转换为对应的中文表达
- technical 参数也要转换为中文，常见翻译：
  * "8k" → "8k"（分辨率单位保留）
  * "Masterpiece" → "杰作"
  * "Highly detailed" → "高度细节" 或 "超精细"
  * "Ray tracing" → "光线追踪"
  * "Ultra detailed" → "超精细"
  * "HDR" → "HDR"（专业术语可保留）
  * "Unreal Engine" → "虚幻引擎"
- raw 字段必须转换为中文版本，即使用户输入的是英文，也要翻译成中文

示例输出格式（原始为英文，全部转换为中文）：
{
  "subject": "赛博朋克风格的雨夜猫咪",
  "art_style": "赛博朋克",
  "visual_elements": ["霓虹灯", "雨", "猫咪", "城市街道"],
  "mood": "阴郁",
  "technical": ["8k", "杰作", "高度细节", "光线追踪"],
  "raw": "赛博朋克风格的雨夜猫咪，霓虹灯，8k 杰作"
}

示例输出格式（原始为中文）：
{
  "subject": "赛博朋克风格的雨夜猫咪",
  "art_style": "赛博朋克",
  "visual_elements": ["霓虹灯", "雨", "猫咪", "城市街道"],
  "mood": "阴郁",
  "technical": ["8k", "杰作", "高度细节", "光线追踪"],
  "raw": "赛博朋克风格的雨夜猫咪，霓虹灯，8k 杰作"
}

现在开始解析用户提供的提示词，并将结果转换为中文。"""
    
    def _parse_with_llm(self, raw_text: str) -> Optional[Dict]:
        """使用 Qwen 3 解析原始文本为结构化 JSON"""
        try:
            # 构造提示词
            user_prompt = f"请解析以下提示词：\n\n{raw_text}"
            
            # 调用 Ollama
            response = self.client.generate(
                prompt=user_prompt,
                system=self.system_prompt,
                temperature=0.3  # 较低温度保证输出稳定
            )
            
            # 清理响应，提取 JSON
            response = response.strip()
            # 移除可能的 markdown 代码块标记
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # 解析 JSON
            parsed = json.loads(response)
            
            # 确保所有必需字段存在
            # raw 字段使用解析后的中文版本，如果没有则使用原始文本（可能是中文）
            raw_chinese = parsed.get("raw", raw_text)
            
            result = {
                "subject": parsed.get("subject", ""),
                "art_style": parsed.get("art_style", ""),
                "visual_elements": parsed.get("visual_elements", []),
                "mood": parsed.get("mood", ""),
                "technical": parsed.get("technical", []),
                "raw": raw_chinese  # 使用中文版本
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始响应: {response[:200]}...")
            return None
        except Exception as e:
            print(f"解析过程出错: {e}")
            return None
    
    def load_jsonl(self, file_path: str) -> List[str]:
        """
        从 JSONL 文件加载数据
        格式: {"file": "文件名", "prompt": "提示词内容"}
        
        Args:
            file_path: JSONL 文件路径
        
        Returns:
            提示词文本列表
        """
        try:
            texts = []
            with jsonlines.open(file_path, mode='r') as reader:
                for item in reader:
                    if isinstance(item, dict) and "prompt" in item:
                        text = item["prompt"]
                        if text and isinstance(text, str) and text.strip():
                            texts.append(text.strip())
            
            print(f"✓ 从 {file_path} 加载了 {len(texts)} 条记录")
            return texts
            
        except Exception as e:
            print(f"✗ 加载 JSONL 失败: {e}")
            return []

    def load_excel(self, file_path: str, sheet_name: str = None, column: str = None) -> List[str]:
        """
        从 Excel 文件加载数据
        
        Args:
            file_path: Excel 文件路径
            sheet_name: 工作表名称（可选，如果为 None 则使用第一个工作表）
            column: 包含提示词的列名（可选，默认第一列）
        
        Returns:
            提示词文本列表
        """
        try:
            # 如果未指定工作表，先检查有哪些工作表
            if sheet_name is None:
                xl_file = pd.ExcelFile(file_path)
                sheet_names = xl_file.sheet_names
                if len(sheet_names) == 1:
                    # 只有一个工作表，直接使用
                    sheet_name = sheet_names[0]
                    print(f"检测到工作表: {sheet_name}")
                elif len(sheet_names) > 1:
                    # 多个工作表，使用第一个
                    sheet_name = sheet_names[0]
                    print(f"检测到多个工作表，使用第一个: {sheet_name}")
                    print(f"可用工作表: {', '.join(sheet_names)}")
            
            # 读取指定工作表
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 检查返回的是字典还是 DataFrame
            if isinstance(df, dict):
                # 如果是字典，使用第一个 DataFrame
                df = list(df.values())[0]
                print(f"检测到多个工作表，使用第一个")
            
            # 如果没有指定列，使用第一列
            if column is None:
                column = df.columns[0]
            
            # 提取文本，过滤空值
            texts = df[column].dropna().astype(str).tolist()
            texts = [t.strip() for t in texts if t.strip()]
            
            print(f"✓ 从 {file_path} 加载了 {len(texts)} 条记录")
            return texts
            
        except Exception as e:
            print(f"✗ 加载 Excel 失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def load_csv(self, file_path: str, column: str = None) -> List[str]:
        """
        从 CSV 文件加载数据
        
        Args:
            file_path: CSV 文件路径
            column: 包含提示词的列名（可选，默认第一列）
        
        Returns:
            提示词文本列表
        """
        try:
            df = pd.read_csv(file_path)
            
            if column is None:
                column = df.columns[0]
            
            texts = df[column].dropna().astype(str).tolist()
            texts = [t.strip() for t in texts if t.strip()]
            
            print(f"✓ 从 {file_path} 加载了 {len(texts)} 条记录")
            return texts
            
        except Exception as e:
            print(f"✗ 加载 CSV 失败: {e}")
            return []
    
    def process_batch(self, texts: List[str], output_path: str = None, append: bool = False) -> str:
        """
        批量处理文本，生成结构化 JSONL 文件
        
        Args:
            texts: 原始文本列表
            output_path: 输出文件路径（可选）
            append: 是否追加模式（True=追加，False=覆盖）
        
        Returns:
            输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(PROCESSED_DATA_DIR, "structured_data.jsonl")
        
        # 如果追加模式，读取现有数据，避免重复
        existing_raws = set()
        if append and os.path.exists(output_path):
            print(f"检测到现有文件，读取已有数据以避免重复...")
            try:
                with jsonlines.open(output_path, mode='r') as reader:
                    for item in reader:
                        raw = item.get('raw', '')
                        if raw:
                            existing_raws.add(raw)
                print(f"  已读取 {len(existing_raws)} 条现有记录")
            except Exception as e:
                print(f"  读取现有文件失败: {e}，将覆盖文件")
                append = False
        
        # 过滤掉已存在的文本
        if existing_raws:
            original_count = len(texts)
            texts = [t for t in texts if t not in existing_raws]
            skipped_count = original_count - len(texts)
            if skipped_count > 0:
                print(f"  跳过 {skipped_count} 条已存在的记录")
        
        if not texts:
            print("\n所有记录都已存在，无需处理")
            return output_path
        
        processed_count = 0
        failed_count = 0
        
        # 根据模式选择写入方式
        mode = 'a' if append else 'w'
        with jsonlines.open(output_path, mode=mode) as writer:
            for text in tqdm(texts, desc="处理中"):
                parsed = self._parse_with_llm(text)
                
                if parsed:
                    writer.write(parsed)
                    processed_count += 1
                else:
                    failed_count += 1
                    # 即使解析失败，也保存原始数据
                    writer.write({
                        "subject": "",
                        "art_style": "",
                        "visual_elements": [],
                        "mood": "",
                        "technical": [],
                        "raw": text
                    })
        
        print(f"\n✓ 处理完成！")
        print(f"  成功: {processed_count} 条")
        print(f"  失败: {failed_count} 条")
        if append:
            print(f"  模式: 追加到现有文件")
        else:
            print(f"  模式: 覆盖文件")
        print(f"  输出文件: {output_path}")
        
        return output_path


if __name__ == "__main__":
    # 测试示例
    pipeline = ETLPipeline()
    
    # 测试连接
    if pipeline.client.test_connection():
        # 示例：处理单个文本
        test_text = "A cyberpunk cat in rainy night, neon lights, 8k masterpiece, highly detailed"
        result = pipeline._parse_with_llm(test_text)
        print("\n解析结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
