import os

import gradio as gr

from dots_ocr.parser import do_parse
from utils.common_utils import get_filename, delete_directory_if_non_empty, get_sorted_md_files
from utils.log_utils import log

# md存储的临时模型
base_md_dir = '/Users/Python/project/project-learn/python-code/Multimodal_RAG/output'


class ProcessorAPP:

    def __init__(self):
        self.pdf_path = None  # 用于记录用户上传的 PDF 文件路径
        self.md_dir = None  # 用于记录 Markdown 文件存储的目录
        self.md_files = None  # 用于记录 MD 文件列表
        self.file_contents: dict[str, str] = {}  # 结构改为：{ "文件名.md": "文件具体内容" }

    def upload_pdf(self, pdf_path):
        """
        处理PDF文件上传的回调函数（由 Gradio 自动触发）
        Args:
            pdf_path (str): PDF 文件路径。Gradio 的 gr.File() 默认传入的是路径字符串
        Returns:
            status: 状态信息
            parse_btn: 解析按钮
        """
        log.info(f"上传PDF文件: {pdf_path}")
        # 如果是空文件则赋值 None，否则赋值为文件路径
        self.pdf_path = pdf_path if pdf_path else None

        if self.pdf_path:
            return [
                f"PDF文件已上传：{os.path.basename(self.pdf_path)}",  # status
                gr.Button(interactive=True)  # 有文件上传后，解析按钮变为可交互状态
            ]

        return [
            "上传文件没有成功，请重新上传PDF文件",  # status
            gr.Button(interactive=False)  # 文件上传失败或者解析失败，解析按钮变为不可交互状态
        ]

    def parse_pdf(self):
        """解析PDF文件，变成多个MD文件，并读取内容到内存中"""

        if not self.pdf_path:
            return [
                "请先上传PDF文件", gr.update(), gr.update(interactive=False), gr.update(interactive=False)
            ]
        log.info("开始解析PDF文件")

        # 根据PDF文件名生成对应的MD存储目录
        md_files_dir = os.path.join(base_md_dir, get_filename(self.pdf_path, False))

        # 清理该目录下已有的旧文件
        delete_directory_if_non_empty(md_files_dir)

        # 调用核心解析函数（依赖外部的dots_ocr模型）
        try:
            do_parse(input_path=self.pdf_path, num_thread=32, no_fitz_preprocess=True)
        except Exception as e:
            log.error(f"解析PDF失败: {e}")
            return [
                "解析PDF文件失败，请检查模型服务是否正常", gr.update(), gr.update(interactive=True), gr.update(
                    interactive=False)
            ]

        # 检查解析结果目录是否成功生成
        if os.path.isdir(md_files_dir):
            self.md_dir = md_files_dir
            self.md_files = get_sorted_md_files(self.md_dir)
            log.info(f"PDF解析完成，生成了 {len(self.md_files)} 个md文件")

            # 读取文件内容，以“文件名”为Key存入字典
            self.file_contents.clear()  # 清空旧数据

            # 读取所有 md 文件内容
            for f in self.md_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        file_name = os.path.basename(f)  # 提取文件名，如 "chunk_1.md"
                        self.file_contents[file_name] = file.read()  # 存为 {"chunk_1.md": "内容..."}
                except Exception as e:
                    print(f"读取文件 {f} 时出错: {e}")
                    self.file_contents[os.path.basename(f)] = f"读取文件内容时出错: {e}"

            # 提取文件名列表，用于填充前端的下拉框
            file_names = list(self.file_contents.keys())

            return [
                f"解析完成，共 {len(self.md_files)} 个MD文件",
                gr.Dropdown(choices=file_names, label="MD文件列表", interactive=True),  # 更新下拉框选项并启用
                gr.Button(interactive=False),  # 禁用“解析PDF”按钮
                gr.update(interactive=True)  # 启用“存入知识库”按钮
            ]
        # 解析失败的处理分支
        return [
            "解析PDF文件时出错，请检查文件路径是否正确",
            gr.Dropdown(choices=[], label="MD文件列表", interactive=False),
            gr.Button(interactive=True),
            gr.update(interactive=False)
        ]

    def select_md_file(self, selected_file):
        """选择MD文件，展示文件内容

        Args:
            selected_file (str): 选中的 MD 文件名
        Returns:
            file_content (str): 选中的 MD 文件内容
        """
        log.info(f"选中的MD文件: {selected_file}")
        # 直接根据文件名从字典取值，大幅简化了原代码的循环匹配逻辑
        if selected_file and selected_file in self.file_contents:
            return self.file_contents[selected_file]

        return "没有找到该文件内容，请重新选择"

    def create_interface(self):
        """创建 Gradio 界面"""

        with gr.Blocks() as app:
            gr.Markdown("### PDF解析与知识库存储和构建")

            # === 第一行：上传与解析 ===
            with gr.Row():
                pdf_upload = gr.File(label="上传PDF文件")  # PDF上传组件，默认行为 type="filepath"，返回文件路径字符串，而不是文件对象（二进制流）。

                parse_btn = gr.Button(value="解析PDF", variant="primary",
                                      interactive=False)  # “解析PDF”按钮。初始状态 interactive=False（禁用），因为还没上传文件

            # === 状态显示 （不可编辑，用于显示系统提示信息） ===
            status = gr.Textbox(label="状态", value="等待操作...", interactive=False)

            # === 第二行：内容展示 ===
            with gr.Row():
                file_dropdown = gr.Dropdown(label="MD文件列表", choices=[],
                                            interactive=False)  # MD文件下拉列表。初始 choices 为空，不可选

                content = gr.Textbox(label="文件内容", lines=20,
                                     interactive=False)  # 文本展示框。展示选中的 MD 文件内容，行高20。初始 interactive=False（禁用），因为还没选择文件

            # === 第三行：存储按钮。（初始禁用，需要解析成功后才允许点击） ===
            save_btn = gr.Button(value="存入知识库", variant="secondary", interactive=False)  # 存入知识库按钮

            # ================= 事件绑定 =================

            # 监听 pdf_upload 组件的 change 事件（用户选择或清除文件时触发）
            pdf_upload.change(
                fn=self.upload_pdf,  # 回调函数
                inputs=pdf_upload,  # 将当前组件（文件）作为参数传给函数的 pdf_path 参数
                outputs=[status, parse_btn]  # 输出组件
            )

            # 点击解析事件：触发 parse_pdf
            parse_btn.click(
                fn=self.parse_pdf,  # 回调函数
                inputs=[],  # 输入组件为空
                outputs=[status, file_dropdown, parse_btn, save_btn]  # 输出组件
            )

            # 下拉框选择事件：触发 select_md_file
            file_dropdown.change(
                fn=self.select_md_file,
                inputs=file_dropdown,
                outputs=[content]
            )
        return app


if __name__ == '__main__':
    app = ProcessorAPP()
    interface = app.create_interface()
    interface.launch()
