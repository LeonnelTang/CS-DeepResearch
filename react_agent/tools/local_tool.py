import ast
import logging
import time
import random
import asyncio
from urllib import request
from typing import List, Dict, Any, Optional
import arxiv
from bs4 import BeautifulSoup
import logging
import requests
import json
import os
from react_agent.tools import BaseTool  # 假设你的基类保存在 base.py 中

logger = logging.getLogger(__name__)


from dotenv import load_dotenv
load_dotenv()


class ArxivSearchEngine(BaseTool):
    """arXiv搜索引擎类，继承自BaseTool"""

    def __init__(self):
        # 初始化父类 BaseTool
        super().__init__(
            name="arxiv_search",
            description="用于在 arXiv 上搜索学术论文，返回论文元数据，包含标题、作者和摘要。参数格式为 {'query': '搜索关键词', 'limit': 返回结果数}"
        )

        # 默认最大结果数
        self.max_results = 10

        # 初始化 Arxiv 客户端
        try:
            self.arxiv_client = arxiv.Client()
            logger.info("Arxiv 搜索 API 初始化成功")
        except Exception as e:
            logger.error(f"Arxiv 搜索 API 初始化失败: {e}")
            self.arxiv_client = None

    def run(self, params: Any) -> str:
        """返回结构化的搜索结果字符串"""
        # 解析参数
        query = ""
        limit = 3
        if isinstance(params, str):
            query = params
        elif isinstance(params, dict):
            query = params.get("query", "")
            limit = params.get("limit", 3)

        if not query:
            return "错误: 缺少搜索查询词 (query)"

        # 运行搜索并获取结果文本
        try:
            contents = self.get_search_contents(query, limit=limit)

            if not contents:
                return "未找到相关结果。"

            # 将列表转换为统一的字符串结构返回
            return "\n\n---\n\n".join(contents)

        except Exception as e:
            logger.error(f"执行工具 run 方法失败: {e}")
            return f"执行搜索时发生错误: {str(e)}"

    def search(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """搜索指定查询词并返回结果"""
        try:
            logger.info(f"搜索查询: {query}")

            # 确定结果数量
            num_results = limit if limit is not None else self.max_results

            if self.arxiv_client:
                return self._search_with_arxiv(query, num_results)
            else:
                logger.warning("Arxiv客户端未初始化，使用模拟结果")

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def _search_with_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """专门用于搜索arXiv论文的方法"""
        try:
            logger.info(f"使用 Arxiv 搜索查询: {query}")

            import arxiv
            # 构建搜索查询
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            print("等待 3 秒，遵守arXiv API 频率限制...")
            time.sleep(3)

            # 执行搜索
            response = self.arxiv_client.results(search)

            # 解析结果
            results = []
            for paper in response:
                result = {
                    'title': paper.title,
                    'url': paper.entry_id,
                    'snippet': paper.summary[:200] + '...' if len(paper.summary) > 200 else paper.summary,
                    'summary': paper.summary,
                    'authors': [author.name for author in paper.authors],
                    'first_author': paper.authors[0].name if paper.authors else '',
                    'published': paper.published.isoformat(),
                    'pdf_url': paper.pdf_url if hasattr(paper, 'pdf_url') else '',
                    'category': paper.primary_category,
                    'categories': [category for category in paper.categories],
                    'source': 'arxiv'
                }

                if result['category'].startswith('cs.'):
                    results.append(result)
                else:
                    logger.info(f"非计算机科学论文，跳过: {result['title']}")

            logger.info(f"Arxiv搜索完成，获得 {len(results)} 个结果")
            return results

        except ImportError:
            logger.error("未安装arxiv包，请运行: pip install arxiv")
            return []
        except Exception as e:
            logger.error(f"arXiv搜索失败: {e}")
            return []

    def get_search_contents(self, query: str, limit: int = 3) -> List[str]:
        """获取搜索结果的内容文本（arxiv）"""
        try:
            # 调用搜索
            search_results = self.search(query, limit=limit)

            if not search_results:
                return []

            # 提取前三个结果的content字段
            contents = []
            for result in search_results[:limit]:
                title = result.get('title', '').strip()
                first_author = result.get('first_author', '').strip()
                summary = result.get('summary', '').strip()

                # 拼接内容作为完整的content
                combined_content = f"{title}\nFirst Author: {first_author}\nSummary: {summary}".strip()
                if combined_content:
                    contents.append(combined_content)

            logger.info(f"搜索查询 '{query}' 获得 {len(contents)} 个内容")
            return contents

        except Exception as e:
            logger.error(f"获取搜索内容失败: {e}")
            return []


class GetRecentAIPapers(BaseTool):
    """通过直接访问 arXiv 网页获取最新 AI 论文的工具类"""

    def __init__(self):
        super().__init__(
            name="get_recent_ai_papers",
            description="获取 arXiv 上计算机科学/人工智能 (cs.AI) 领域的最新学术论文。参数格式为 {'limit': 返回论文数量}，默认返回5篇。"
        )

    def run(self, params: Any) -> str:
        """核心执行方法，抓取网页并解析 HTML 结构"""
        # 1. 解析参数
        limit = 5
        if isinstance(params, dict):
            limit = params.get("limit", 5)
        elif isinstance(params, int):
            limit = params
        elif isinstance(params, str) and params.isdigit():
            limit = int(params)

        url = "https://arxiv.org/list/cs.AI/recent"

        try:
            logger.info(f"正在抓取 {url}...")

            # 2. 发送请求 (添加 User-Agent 伪装成浏览器，防止被 arXiv 屏蔽)
            req = request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with request.urlopen(req) as response:
                html = response.read()

            # 3. 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(html, 'html.parser')

            # arXiv 的列表数据存放在 <dl> 标签下：
            # <dt> 存放论文的链接标识
            # <dd> 存放论文的标题和作者信息
            dts = soup.find_all('dt')
            dds = soup.find_all('dd')

            if not dts or not dds:
                return "错误: 无法解析页面结构，arXiv 网页的 HTML 可能已发生更改。"

            formatted_results = []

            # 4. 遍历提取信息
            for i in range(min(limit, len(dts))):
                dt = dts[i]
                dd = dds[i]

                # 提取 Paper ID 和 URL
                list_identifier = dt.find('a', title='Abstract')
                paper_id = list_identifier.text.strip() if list_identifier else "未知 ID"
                paper_url = f"https://arxiv.org/abs/{paper_id}" if paper_id != "未知 ID" else "未知链接"

                # 提取标题
                title_div = dd.find('div', class_='list-title')
                if title_div:
                    # 移除 "Title: " 前缀并清理多余空白符
                    title = title_div.text.replace('Title:', '').strip()
                else:
                    title = "未获取到标题"

                # 提取作者
                authors_div = dd.find('div', class_='list-authors')
                if authors_div:
                    author_links = authors_div.find_all('a')
                    authors = ", ".join([a.text for a in author_links])
                else:
                    authors = "未获取到作者"

                # 组装单篇论文文本 (注意：列表页默认不带摘要 summary)
                paper_str = (
                    f"Title: {title}\n"
                    f"Authors: {authors}\n"
                    f"URL: {paper_url}"
                )
                formatted_results.append(paper_str)

            if not formatted_results:
                return "未能从页面提取到任何论文。"

            logger.info(f"成功抓取 {len(formatted_results)} 篇最新 AI 论文")
            return "\n\n---\n\n".join(formatted_results)

        except ImportError:
            return "错误: 未安装 bs4 包。请运行 'pip install beautifulsoup4' 安装 HTML 解析库。"
        except Exception as e:
            logger.error(f"抓取或解析 arXiv 网页失败: {e}")
            return f"获取最新 AI 论文时发生错误: {str(e)}"




class ListSkillsTool(BaseTool):
    """工具：列出当前系统支持的所有技能及描述"""

    def __init__(self, skills_file_path: str = "skills.json"):
        # 满足基类的描述要求：功能描述 + 参数格式
        description = "获取所有可用的技能（Skills）列表及其功能描述。参数格式为 {}"
        super().__init__(name="list_skills", description=description)
        self.skills_file_path = os.path.abspath(skills_file_path)

    def run(self, params: Any = None) -> str:
        """读取本地文件，返回所有技能的概览"""
        if not os.path.exists(self.skills_file_path):
            return "Error: 技能配置文件不存在，请检查路径。"

        try:
            with open(self.skills_file_path, 'r', encoding='utf-8') as f:
                skills_data = json.load(f)

            if not skills_data:
                return "Observation: 当前没有可用的技能。"

            result_lines = ["当前可用的技能列表如下："]
            for skill_name, skill_info in skills_data.items():
                desc = skill_info.get("description", "无描述")
                result_lines.append(f"- {skill_name}: {desc}")

            return "\n".join(result_lines)

        except Exception as e:
            return f"Error: 读取技能列表时发生错误: {str(e)}"


class GetSkillPromptTool(BaseTool):
    """工具：根据技能名称获取对应的工作流程 Prompt"""

    def __init__(self, skills_file_path: str = "skills.json"):
        description = "获取某一具体技能的详细 Prompt（工作流程提示）。参数格式为 {'skill_name': '需要获取的技能名称'}"
        super().__init__(name="get_skill_prompt", description=description)
        self.skills_file_path = skills_file_path

    def run(self, params: Any) -> str:
        """解析参数，返回特定的 Prompt"""
        # 1. 解析和校验参数
        if not isinstance(params, dict) or "skill_name" not in params:
            return "Error: 参数格式错误，必须包含 'skill_name' 键。例如: {'skill_name': 'data_analyst'}"

        skill_name = params["skill_name"]

        # 2. 读取本地 JSON 库
        if not os.path.exists(self.skills_file_path):
            return "Error: 技能配置文件不存在，请检查路径。"

        try:
            with open(self.skills_file_path, 'r', encoding='utf-8') as f:
                skills_data = json.load(f)

            # 3. 匹配并返回对应的 Prompt
            if skill_name in skills_data:
                prompt_content = skills_data[skill_name].get("prompt", "")
                return f"Observation: 成功获取技能 '{skill_name}' 的 Prompt:\n{prompt_content}"
            else:
                return f"Observation: 未找到名为 '{skill_name}' 的技能，请先使用 list_skills 工具查看可用列表。"

        except Exception as e:
            return f"Error: 读取技能 Prompt 时发生错误: {str(e)}"


class JinaReaderTool(BaseTool):
    """工具：使用 Jina Reader API 读取并提取指定 URL 的网页内容"""

    def __init__(self, api_key: str = None):
        super().__init__(
            name="jina_reader",
            description="读取并提取指定 URL 的网页文本内容，将其转换为适合 LLM 阅读的格式。参数格式为 {'url': '目标网址'}"
        )
        # 优先使用传入的 api_key，否则尝试从环境变量获取
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")

    def run(self, params: Any) -> str:
        """核心执行方法，发送 GET 请求读取网页内容"""
        # 1. 解析参数
        url = ""
        if isinstance(params, dict):
            url = params.get("url", "")
        elif isinstance(params, str):
            try:
                parsed_params = json.loads(params)
                if isinstance(parsed_params, dict):
                    url = parsed_params.get("url", "")
                else:
                    url = params
            except json.JSONDecodeError:
                url = params

        if not url:
            return "Error: 缺少必需的参数 'url'"

        url = url.strip()

        # 2. 构建请求
        target_url = f"https://r.jina.ai/{url}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 3. 发送请求并返回结果
        try:
            logger.info(f"正在使用 Jina Reader 读取 URL: {url}")
            response = requests.get(target_url, headers=headers, timeout=15)

            # 检查响应状态码
            if response.status_code == 200:
                return response.text
            else:
                return f"Error: 读取网页失败，状态码 {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            logger.error(f"Jina Reader 网络请求失败: {e}")
            return f"Error: 请求网页时发生网络异常: {str(e)}"
        except Exception as e:
            logger.error(f"Jina Reader 执行失败: {e}")
            return f"Error: 执行读取时发生未知错误: {str(e)}"


class JinaSearchTool(BaseTool):
    """工具：使用 Jina Search API 在网络上进行搜索并返回结果"""

    def __init__(self, api_key: str = None):
        super().__init__(
            name="jina_search",
            description="在网络上搜索信息并返回搜索引擎结果页面 (SERP) 的文本内容。参数格式为 {'query': '搜索关键词'}"
        )
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")

    def run(self, params: Any) -> str:
        """核心执行方法，发送 GET 请求进行搜索"""
        # 1. 解析参数
        query = ""
        if isinstance(params, dict):
            query = params.get("query", "")
        elif isinstance(params, str):
            params_str = params.strip()
            logger.info(f"正在使用 Jina Search 解析参数: {params_str}")

            try:
                # 第一步：尝试标准的 JSON 解析（期望双引号）
                parsed_params = json.loads(params_str)
                if isinstance(parsed_params, dict):
                    query = parsed_params.get('query', '')
                else:
                    query = params_str

            except json.JSONDecodeError:
                try:
                    # 第二步：如果标准 JSON 解析失败，尝试解析 Python 字典格式（处理单引号情况）
                    parsed_params = ast.literal_eval(params_str)
                    if isinstance(parsed_params, dict):
                        query = parsed_params.get('query', '')
                    else:
                        query = params_str

                except (ValueError, SyntaxError):
                    # 第三步：如果既不是 JSON 也不是字典格式，说明大模型真的是直接传了纯文本
                    query = params_str

        logger.info(f"最终提取到的查询词: {query}")


        if not query:
            return "Error: 缺少必需的参数 'query'"

        query = query.strip()

        # 2. 构建请求
        target_url = "https://s.jina.ai/"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        headers["X-Respond-With"] = "no-content"

        # 注意: 示例代码中包含 "X-Respond-With": "no-content"
        # 如果携带此 header，API 可能不会返回具体内容。
        # 作为给大模型提供的工具，我们需要搜索结果的具体文本，因此这里注释掉该 header。
        # 若需要纯粹的 JSON 格式或特定行为，可以按需取消注释或修改。
        # headers["X-Respond-With"] = "no-content"

        # 3. 发送请求并返回结果
        try:
            logger.info(f"正在使用 Jina Search 搜索关键词: {query}")
            # 使用 params 传递查询参数自动处理 URL 编码
            response = requests.get(target_url, params={"q": query}, headers=headers, timeout=15)

            if response.status_code == 200:
                logger.info(f"Jina Search 搜索成功，正在返回结果...")
                return response.text
            else:
                return f"Error: 搜索失败，状态码 {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            logger.error(f"Jina Search 网络请求失败: {e}")
            return f"Error: 搜索时发生网络异常: {str(e)}"
        except Exception as e:
            logger.error(f"Jina Search 执行失败: {e}")
            return f"Error: 执行搜索时发生未知错误: {str(e)}"

# ==========================================
# 测试代码 (Test Cases)
# ==========================================
if __name__ == "__main__":
    # 为了测试方便，你可以在这里设置一个假的或真实的 API Key
    # os.environ["JINA_API_KEY"] = "your_jina_api_key_here"


    # 2. 测试 Jina Search
    search_tool = JinaSearchTool()
    print("--- 测试 JinaSearchTool ---")
    print(search_tool.to_str())
    # 填入搜索关键词进行测试
    search_result = search_tool.run({"query": "2025年"})
    print(f"Search 结果前 200 个字符: \n{search_result}...")