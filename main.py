# main.py
from vector_store.chroma_store import ChromaVectorStore
from llm.qwen_llm import QwenLLM


# === 1. 准备初始文档 ===
INITIAL_DOCS = [
    "中电金信软件有限公司（简称“中电金信”）是中国电子信息产业集团有限公司（CEC，中国电子）旗下专注于金融科技领域的高科技企业。",
    "Chroma 是一个开源的向量数据库，专为 AI 应用设计，支持嵌入、检索和存储。",
    "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的技术，能显著提升问答系统的准确性。",
    "DashScope 是阿里云提供的模型开放平台，支持 Qwen 系列模型的 API 调用。"
]

def main():
    print("🚀 初始化 RAG 系统...\n")

    # === 2. 初始化向量数据库 ===
    vector_store = ChromaVectorStore(collection_name="demo_rag")
    vector_store.add_documents(INITIAL_DOCS)
    print(f"✅ 已将 {len(INITIAL_DOCS)} 篇文档存入 Chroma 向量库\n")

    # === 3. 初始化 LLM ===
    llm = QwenLLM()

    # === 4. 用户提问 ===
    query = "中电金信 是什么？"

    print(f"❓ 用户问题: {query}\n")

    # === 5. 检索相关文档 ===
    retrieved_docs = vector_store.similarity_search(query, k=2)
    context = "\n".join(retrieved_docs)
    print("🔍 检索到的相关上下文:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  [{i}] {doc}")
    print()

    # === 6. 构造 Prompt 并生成答案 ===
    prompt = f"""你是一个智能助手，请根据以下上下文回答问题。如果上下文不足以回答，请说“我不知道”。

上下文：
{context}

问题：{query}
回答："""

    answer = llm.generate(prompt)
    print("🤖 Qwen 回答:")
    print(answer)

if __name__ == "__main__":
    main()