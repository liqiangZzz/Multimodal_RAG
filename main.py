import gradio as gr

# md存储的临时模型
base_md_dir = '/Users/Python/project/project-learn/python-code/Multimodal_RAG/output'


class ProcessorAPP:

    def __init__(self):
        self.pdf_path = None  # PDF文件路径
        self.md_dir = None  # MD文件存储目录
        self.md_files = None  # MD文件列表

    def create_interface(self):
        """创建一个构建多模态知识库的界面"""

        with gr.Blocks() as app:
            gr.Markdown("### PDF解析与知识库存储和构建")

            with gr.Row():
                pdf_upload = gr.File(label="上传PDF文件")  # PDF上传组件

                parse_btn = gr.Button(value="解析PDF", variant="primary", interactive=False)  # 解析PDF按钮

            status = gr.Textbox(label="状态", value="等待操作...", interactive=False)  # 状态显示组件

            with gr.Row():
                md_files = gr.Dropdown(label="MD文件列表", choices=[], interactive=False)  # MD文件列表组件

                md_content = gr.Textbox(label="文件内容", lines=20, interactive=False)  # 文件内容组件(展示选中文件的内容)

            save_btn = gr.Button(value="存入知识库", variant="secondary", interactive=False)  # 存入知识库按钮
        return app


if __name__ == '__main__':
    app = ProcessorAPP()
    interface = app.create_interface()
    interface.launch()
