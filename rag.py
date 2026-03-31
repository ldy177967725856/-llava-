import gradio as gr
from pathlib import Path
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
import shutil

# --- 1. 全局配置与资源初始化 ---
# 存储向量数据和临时上传文件的目录
STORAGE_DIR = Path("tmp/gradio_rag_storage")
UPLOAD_DIR = STORAGE_DIR / "uploads"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 定义本地嵌入模型 (Ollama)
local_embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=768)

# 定义向量数据库 (LanceDB)
vector_db = LanceDb(
    table_name="gradio_docs",
    uri=str(STORAGE_DIR / "lancedb"),
    search_type=SearchType.vector,
    embedder=local_embedder,
)

# 临时知识库对象，用于后续动态加载
knowledge_base = PDFKnowledgeBase(path=UPLOAD_DIR, vector_db=vector_db)


# --- 2. 后端核心逻辑函数 ---
def process_active_rag(image_path, doc_file, user_question):
    """
    处理 Gradio 界面提交的 RAG 请求。
    1. 保存文档 -> 2. 更新向量库 -> 3. 调用多模态 Agent -> 4. 生成回答
    """
    if not user_question:
        return "⚠️ 请输入你的问题。"

    system_prompt_instruction = "你是一个全能的本地分析助手。"

    # --- 步骤 A: 动态处理文档 (RAG) ---
    if doc_file is not None:
        # 清空之前的上传，保证只针对当前文件进行 RAG
        for f in UPLOAD_DIR.glob("*"):
            if f.is_file(): f.unlink()
        
        # 将 gradio 传入的临时文件复制到指定的上传目录
        uploaded_path = UPLOAD_DIR / Path(doc_file.name).name
        shutil.copy(doc_file.name, uploaded_path)
        
        gr.Info("⏳ 正在分析文档内容并建立索引...")
        # 强制更新知识库 (upsert 会重置表并重新加载，适合单文件场景)
        knowledge_base.load(upsert=True)
        
        system_prompt_instruction += "你可以从提供的【知识库】中检索文本信息。"
    else:
        # 如果没有传文件，确保知识库是空的，防止误用之前的数据
        try: vector_db.drop_table()
        except: pass

    # --- 步骤 B: 初始化 Agent (假设后端支持多模态) ---
    agent = Agent(
        model=OpenAIChat(
            id="llava-multimodal", # 名字随便取，关键是 base_url
            base_url="http://localhost:8080/v1", # llama.cpp server
            api_key="sk-no-key-required"
        ),
        knowledge=knowledge_base if doc_file is not None else None,
        search_knowledge=True if doc_file is not None else False,
        markdown=True,
        instructions=[
            system_prompt_instruction,
            "1. 如果用户上传了图片，首先描述和分析图片中的关键图形信息。",
            "2. 如果用户上传了文档，结合检索到的文档知识回答问题。",
            "3. 如果文档中找不到相关内容，请明确告知，绝不能编造事实。",
            "4. 最终回答应逻辑清晰，综合图片和文档的信息（如果都有）。"
        ],
    )

    # --- 步骤 C: 执行生成 (关键点：传入图片路径列表) ---
    images_to_send = [image_path] if image_path else None
    
    if image_path is not None:
        user_question = "<image>\n" + user_question


    try:
        gr.Info("🤖 大模型正在思考和组织回答...")
        # 【修改这里】使用 agent.run() 获取真实的返回对象
        response = agent.run(user_question, images=images_to_send)
        # 获取纯文本内容
        return response.content
    except Exception as e:
        return f"❌ 发生错误: {str(e)}"


# --- 3. 定义 Gradio 可视化界面 ---
with gr.Blocks(title="本地智能 RAG 助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 本地智能 RAG 分析助手")
    gr.Markdown("同时支持 **图像图形分析** 和 **PDF 文档检索**，完全基于本地模型运行。")

    with gr.Row():
        # 左侧输入面板
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入数据")
            input_image = gr.Image(label="上传图形/图片 (用于图形分析)", type="filepath")
            input_doc = gr.File(label="上传知识库文件 (仅限 PDF)", file_types=[".pdf"])
            input_query = gr.Textbox(label="输入你的问题", placeholder="例如：请分析图中的趋势，并结合文档第3页的内容...")
            
            with gr.Row():
                clear_btn = gr.Button("清空")
                submit_btn = gr.Button("开始分析", variant="primary")

        # 右侧输出面板
        with gr.Column(scale=2):
            gr.Markdown("### 2. 分析结果")
            output_answer = gr.Markdown(label="AI 的回答")

    # 定义交互事件
    submit_btn.click(
        fn=process_active_rag,
        inputs=[input_image, input_doc, input_query],
        outputs=output_answer
    )
    
    # 清空逻辑
    clear_btn.click(lambda: (None, None, "", ""), outputs=[input_image, input_doc, input_query, output_answer])

# --- 4. 启动界面 ---
if __name__ == "__main__":
    # 第一次运行建议先创建表
    print("⏳ 初始化 LanceDB 表...")
    try:
        vector_db.create_table()
    except Exception as e:
        print(f"Table might exist or error: {e}")

    demo.launch(server_port=7860, share=False)