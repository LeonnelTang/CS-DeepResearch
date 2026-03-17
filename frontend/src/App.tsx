import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import './index.css';

// 定义数据结构
interface StepData {
  step: number;
  think: string;
  action: string;
  observation: string;
}

interface ToolOption {
  id: number;
  name: string;
  type: string;
  description: string;
}

// 独立的步骤卡片组件，自带折叠状态管理
const StepCard = ({ stepData }: { stepData: StepData }) => {
  // 默认展开新卡片
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="shrink-0 bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden transition-all hover:shadow-md">
      {/* 💡 卡片头部：点击可折叠/展开 */}
      <div
        className="bg-gray-50 hover:bg-gray-100 border-b px-4 py-3 flex items-center justify-between cursor-pointer transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-3">
          <span className="font-bold text-sm text-gray-700 bg-white px-2 py-1 rounded shadow-sm border border-gray-200">
            Step {stepData.step}
          </span>
          {/* 折叠时，巧妙地展示 Action 的名字，方便概览 */}
          {!isOpen && (
            <span className="text-xs text-gray-500 font-mono bg-gray-200/50 px-2 py-1 rounded line-clamp-1 max-w-md">
              {stepData.action}
            </span>
          )}
        </div>

        {/* 动态箭头图标 */}
        <button className="text-gray-400 hover:text-gray-600 focus:outline-none">
          <svg
            className={`w-5 h-5 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* 💡 卡片主体：根据 isOpen 决定是否渲染 */}
      {isOpen && (
        <div className="p-4 space-y-4 animate-fade-in-down">
          {/* 思考 (Think) */}
          <div className="flex gap-3">
            <div className="text-xl">🧠</div>
            <div className="flex-1 bg-blue-50/50 rounded-lg p-3 text-sm text-gray-700 italic border border-blue-100/50">
              <ReactMarkdown>{stepData.think}</ReactMarkdown>
            </div>
          </div>

          {/* 动作 (Action) */}
          <div className="flex gap-3 items-center">
            <div className="text-xl">🛠️</div>
            <code className="flex-1 bg-gray-100 rounded-md p-2 text-sm text-pink-600 font-mono border border-gray-200">
              {stepData.action}
            </code>
          </div>

          {/* 观察 (Observation) */}
          <div className="flex gap-3">
            <div className="text-xl">👁️</div>
            <div className="flex-1 bg-green-50/50 border border-green-100/50 rounded-lg p-3 text-sm text-gray-800 overflow-x-auto max-h-60 overflow-y-auto scrollbar-thin">
              <pre className="whitespace-pre-wrap font-sans text-xs text-gray-600 leading-relaxed">
                {stepData.observation}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default function App() {
  // === 核心状态 ===
  const [query, setQuery] = useState("请调研强化学习中 GRPO 算法的最新进展，并对比它与 PPO 的异同。");
  const [isRunning, setIsRunning] = useState(false);
  const [steps, setSteps] = useState<StepData[]>([]);
  const [liveReport, setLiveReport] = useState("Waiting for the agent to start...");
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // === 工具选择状态 ===
  const [availableTools, setAvailableTools] = useState<ToolOption[]>([]);
  const [selectedTools, setSelectedTools] = useState<number[]>([]);

  const centerPanelRef = useRef<HTMLDivElement>(null);

  // 初始化时：从后端拉取可用工具列表
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/tools")
      .then(res => res.json())
      .then(data => {
        if (data.tools) {
          setAvailableTools(data.tools);
          // 默认全不选
          setSelectedTools([]);
        }
      })
      .catch(err => {
        console.error("无法获取工具列表:", err);
        setError("无法连接到后端获取工具列表，请检查 FastAPI 是否运行。");
      });
  }, []);

  // 每次有新步骤时，中间面板自动滚动到底部
  useEffect(() => {
    if (centerPanelRef.current) {
      centerPanelRef.current.scrollTop = centerPanelRef.current.scrollHeight;
    }
  }, [steps, finalAnswer]);

  // 处理工具勾选/取消勾选
  const handleToolToggle = (id: number) => {
    setSelectedTools(prev =>
      prev.includes(id) ? prev.filter(tId => tId !== id) : [...prev, id]
    );
  };

  const startResearch = () => {
    if (!query.trim()) return;
    if (selectedTools.length === 0) {
      setError("请至少选择一个工具！");
      return;
    }

    setSteps([]);
    setLiveReport("Initializing IterResearch Workspace...");
    setFinalAnswer(null);
    setError(null);
    setIsRunning(true);

    // 拼接 tools_idx 参数 (例如: "0,1,3")
    const toolsIdxParams = selectedTools.join(",");
    const url = `http://127.0.0.1:8000/api/research?query=${encodeURIComponent(query)}&tools_idx=${toolsIdxParams}`;

    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'step_update') {
          if (data.report) setLiveReport(data.report);
          setSteps((prevSteps) => {
            const existingStepIndex = prevSteps.findIndex(s => s.step === data.step);
            if (existingStepIndex >= 0) {
              const newSteps = [...prevSteps];
              newSteps[existingStepIndex] = { ...data };
              return newSteps;
            } else {
              return [...prevSteps, { ...data }];
            }
          });
        } else if (data.type === 'finish') {
          setFinalAnswer(data.answer);
          setIsRunning(false);
          eventSource.close();
        }
      } catch (err) {
        console.error("解析流数据失败:", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE 连接错误:", err);
      setError("与服务器的连接中断，请检查后端是否在 8000 端口运行。");
      setIsRunning(false);
      eventSource.close();
    };
  };

  const markdownComponents = {
    // ... 保持原有 markdown 渲染配置不变
    h1: ({node, ...props}: any) => <h1 className="text-2xl font-bold mt-4 mb-2" {...props} />,
    h2: ({node, ...props}: any) => <h2 className="text-xl font-bold mt-4 mb-2" {...props} />,
    h3: ({node, ...props}: any) => <h3 className="text-lg font-bold mt-3 mb-2" {...props} />,
    p: ({node, ...props}: any) => <p className="mb-3 leading-relaxed" {...props} />,
    ul: ({node, ...props}: any) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
    ol: ({node, ...props}: any) => <ol className="list-decimal pl-5 mb-3 space-y-1" {...props} />,
    li: ({node, ...props}: any) => <li className="mb-1" {...props} />,
    a: ({node, ...props}: any) => <a className="text-blue-600 hover:underline" target="_blank" rel="noreferrer" {...props} />,
    blockquote: ({node, ...props}: any) => <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-600 my-4" {...props} />,
  };

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans overflow-hidden">

      {/* ================= 左侧栏：控制台 ================= */}
      <div className="w-1/4 bg-white border-r flex flex-col shadow-sm z-10">
        <div className="p-6 border-b">
          <h1 className="text-xl font-black text-indigo-700 tracking-tight flex items-center gap-2">
            <span>🔬</span> IterResearch
          </h1>
          <p className="text-xs text-gray-500 mt-1">Autonomous Deep Research Agent</p>
        </div>

        <div className="p-6 flex-1 flex flex-col gap-6 overflow-y-auto">
          {/* 输入框区域 */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Research Question</label>
            <textarea
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all resize-none h-32 text-sm"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe what you want to research..."
              disabled={isRunning}
            />
          </div>

          {/* 工具选择器区域 */}
          <div className="flex-1 flex flex-col min-h-[300px]">
            <label className="block text-sm font-semibold text-gray-700 mb-2 flex justify-between items-center">
              <span>Enabled Tools</span>
              <span className="text-xs font-normal text-gray-500">{selectedTools.length} selected</span>
            </label>

            {/* 💡 修复：去掉了 max-h-48，换成了 flex-1 和更宽松的高度限制，列表会变得很长且舒适 */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-3 flex-1 overflow-y-auto">
              {availableTools.length === 0 ? (
                <p className="text-xs text-gray-400 text-center py-4">加载工具中...</p>
              ) : (
                availableTools.map((tool, index) => {
                  // 💡 修复：这里使用真实的 index 作为唯一绑定值
                  const isChecked = selectedTools.includes(index);

                  return (
                    <label
                      key={index}
                      className={`flex items-start gap-3 p-3 rounded-md cursor-pointer transition-all border ${
                        isChecked 
                          ? 'bg-indigo-50 border-indigo-200 shadow-sm' 
                          : 'hover:bg-gray-100 border-transparent'
                      }`}
                    >
                      <input
                        type="checkbox"
                        className="mt-1 w-4 h-4 text-indigo-600 rounded border-gray-300 focus:ring-indigo-500 cursor-pointer"
                        checked={isChecked}
                        onChange={() => handleToolToggle(index)}
                        disabled={isRunning}
                      />
                      <div className="flex flex-col">
                        <span className="text-sm font-bold text-gray-800 flex items-center gap-1.5">
                          {tool.type === 'local' ? '📍' : '🌐'} {tool.name}
                        </span>
                        <span className="text-xs text-gray-500 mt-1 leading-relaxed line-clamp-2" title={tool.description}>
                          {tool.description}
                        </span>
                      </div>
                    </label>
                  );
                })
              )}
            </div>
          </div>

          {/* 错误提示 */}
          {error && (
            <div className="p-3 bg-red-50 text-red-600 border border-red-200 rounded-lg text-sm">
              {error}
            </div>
          )}

          {/* 启动按钮 */}
          <button
            onClick={startResearch}
            disabled={isRunning || !query.trim() || selectedTools.length === 0}
            className={`py-3 px-4 rounded-lg font-bold text-white transition-all flex justify-center items-center gap-2 mt-auto ${
              (isRunning || !query.trim() || selectedTools.length === 0)
                ? 'bg-indigo-400 cursor-not-allowed' 
                : 'bg-indigo-600 hover:bg-indigo-700 active:scale-95 shadow-md hover:shadow-lg'
            }`}
          >
            {isRunning ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Researching...
              </>
            ) : 'Start Deep Research'}
          </button>
        </div>
      </div>

      {/* ================= 中间区域：执行流与结果 ================= */}
      <div className="flex-1 flex flex-col relative bg-[#f9fafb] min-w-0" ref={centerPanelRef}>
        {/* ... (此处代码与之前完全一致) ... */}
        {/* 为了保持回复精简，这部分 UI 不变，已包含在完整代码中 */}
        <div className="sticky top-0 bg-white/80 backdrop-blur-md border-b px-8 py-4 z-10 shadow-sm">
          <h2 className="text-lg font-bold text-gray-800">Execution Trajectory</h2>
        </div>

        <div className="p-8 pb-32 flex flex-col gap-6 overflow-y-auto h-full">
          {/* 渲染历史步骤卡片 */}
          {steps.map((s, idx) => (
              <StepCard key={idx} stepData={s}/>
          ))}

          {finalAnswer && (
              <div
                  className="shrink-0 bg-white border-2 border-indigo-200 rounded-xl shadow-lg mt-4 overflow-hidden animate-fade-in-up">
                <div className="bg-indigo-50 border-b border-indigo-100 px-6 py-4 flex items-center gap-2">
                  <span className="text-2xl">🎉</span>
                  <h2 className="text-xl font-black text-indigo-800">Final Research Report</h2>
                </div>
                <div className="p-8 prose prose-indigo max-w-none text-gray-800">
                  <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                      components={markdownComponents}
                  >
                    {finalAnswer}
                  </ReactMarkdown>
                </div>
              </div>
          )}
        </div>
      </div>

      {/* ================= 右侧栏：MDP 演进报告 (核心设计) ================= */}
      <div className="w-1/3 bg-white border-l shadow-inner flex flex-col z-10">
        <div className="p-4 border-b bg-gray-50 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl">📝</span>
            <h2 className="font-bold text-gray-800">Global Workspace</h2>
          </div>
          <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-semibold animate-pulse">
            Live Status
          </span>
        </div>

        <div className="flex-1 p-6 overflow-y-auto bg-white">
          <div className="prose prose-sm max-w-none text-gray-700">
            {liveReport === "Waiting for the agent to start..." ? (
               <div className="flex flex-col items-center justify-center h-full text-gray-400 mt-20">
                 <span className="text-4xl mb-4">🗂️</span>
                 <p>The evolving report will appear here.</p>
               </div>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                components={markdownComponents}
              >
                {liveReport}
              </ReactMarkdown>
            )}
          </div>
        </div>
      </div>

    </div>
  );
}