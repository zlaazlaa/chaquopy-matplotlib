import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug

WORKING_DIR = "./dickens"
from com.chaquo.python import Python
files_dir = str(Python.getPlatform().getApplication().getFilesDir())
WORKING_DIR = os.path.join(files_dir, WORKING_DIR)


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    from com.chaquo.python import Python
    log_dir = str(Python.getPlatform().getApplication().getFilesDir())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "qwen-max-latest",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-31f865bf94244a55b72788523168abd9",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs
    )

import numpy as np
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-v4",
        api_key="sk-31f865bf94244a55b72788523168abd9",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=embedding_func
                ),
        llm_model_func=llm_model_func,
        embedding_batch_num=10
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main(logs=None):
    # logs: 用于收集关键信息的列表
    if logs is None:
        logs = []

    rag = None
#     # 检查 OPENAI_API_KEY 环境变量是否存在，不存在则硬编码设置
#     if not os.getenv("OPENAI_API_KEY"):
#         os.environ["OPENAI_API_KEY"] = "sk-JY6HYn7lyyznGdh5efNVFdhqDO7SpEfAiPVPECaJNQHH5BnR"  # 替换为你的实际 key
#         logs.append("未检测到 OPENAI_API_KEY，已使用硬编码默认值。")

    try:
        # 清理旧数据文件
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logs.append(f"已删除旧文件: {file_path}")

        # 初始化 RAG 实例
        rag = await initialize_rag()

        # 测试 embedding 函数
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        logs.append("=======================")
        logs.append("Test embedding function")
        logs.append("=======================")
        logs.append(f"Test dict: {test_text}")
        logs.append(f"Detected embedding dimension: {embedding_dim}")
        logs.append(f"Embedding result: {embedding}")

        from com.chaquo.python import Python
        files_dir = str(Python.getPlatform().getApplication().getFilesDir())
        book_path = os.path.join(files_dir, "book.txt")

        from book_txt import book
        with open(book_path, "w", encoding="utf-8") as f:
            f.write(book)
        with open(book_path, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # naive 查询
        logs.append("=====================")
        logs.append("Query mode: naive")
        logs.append("=====================")
        result_naive = await rag.aquery(
            "这个故事的主要主题是什么？", param=QueryParam(mode="naive")
        )
        logs.append(str(result_naive))

        # local 查询
        logs.append("=====================")
        logs.append("Query mode: local")
        logs.append("=====================")
        result_local = await rag.aquery(
            "这个故事的主要主题是什么？", param=QueryParam(mode="local")
        )
        logs.append(str(result_local))

        # global 查询
        logs.append("=====================")
        logs.append("Query mode: global")
        logs.append("=====================")
        result_global = await rag.aquery(
            "这个故事的主要主题是什么？",
            param=QueryParam(mode="global"),
        )
        logs.append(str(result_global))

        # hybrid 查询
        logs.append("=====================")
        logs.append("Query mode: hybrid")
        logs.append("=====================")
        result_hybrid = await rag.aquery(
            "这个故事的主要主题是什么？",
            param=QueryParam(mode="hybrid"),
        )
        logs.append(str(result_hybrid))

    except Exception as e:
        logs.append(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()
    return "\n".join(logs)


def run_main_sync():
    import asyncio
    import yappi # 导入 yappi

    configure_logging()

    yappi.set_clock_type("wall") # 关键！设置为测量墙上时钟时间
    yappi.start() # 开始分析

    logs = []
    result = asyncio.run(main(logs))

    yappi.stop() # 停止分析

    print("=" * 50)
    print("Yappi Stats (Wall Time):")
    # 打印所有线程/协程的统计信息
    # 按总耗时 (ttot) 排序
    stats = yappi.get_func_stats()
    stats.sort("ttot", "desc")
    stats.print_all(
        columns={
            0: ("name", 80),
            1: ("ncall", 5),
            2: ("tsub", 8),
            3: ("ttot", 8),
            4: ("tavg", 8),
        }
    )
    print("=" * 50)

    # 生成可用于火焰图的文件 (callgrind 格式)
    profile_file = "run_main_sync11.callgrind"
    # 注意：需要找到一个能适配您环境的路径
    from com.chaquo.python import Python
    files_dir = str(Python.getPlatform().getApplication().getFilesDir())
    profile_file = os.path.join(files_dir, profile_file)

    stats.save(profile_file, type='callgrind')
    print(f"\n性能分析数据已保存到: {profile_file}")
    print("您可以使用 kcachegrind 或其他兼容工具查看此文件，或将其转换为火焰图。")

    return result

if __name__ == "__main__":
    result = run_main_sync()
    print(result)
    print("\nDone!")