import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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

// 独立的步骤卡片组件 (Apple 悬浮卡片风格)
const StepCard = ({ stepData }: { stepData: StepData }) => {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="shrink-0 bg-white rounded-2xl shadow-[0_4px_20px_rgba(0,0,0,0.04)] border border-gray-100 overflow-hidden transition-all duration-300">
      {/* 💡 卡片头部：极简的灰度背景与分隔 */}
      <div
        className="bg-white hover:bg-gray-50/80 border-b border-gray-100 px-5 py-3.5 flex items-center justify-between cursor-pointer transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold text-[13px] text-[#1D1D1F] bg-[#F2F2F7] px-2.5 py-1 rounded-md">
            Step {stepData.step}
          </span>
          {!isOpen && (
            <span className="text-[13px] text-[#86868B] font-mono line-clamp-1 max-w-md">
              {stepData.action}
            </span>
          )}
        </div>

        {/* 极简箭头图标 */}
        <button className="text-[#86868B] hover:text-[#1D1D1F] focus:outline-none">
          <svg
            className={`w-4 h-4 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* 💡 卡片主体：留白与区块分割 */}
      {isOpen && (
        <div className="p-6 space-y-6 animate-fade-in-down">
          {/* 思考 (Think) - 弱化图标，强调文字 */}
          <div className="flex gap-4">
            <div className="mt-0.5 flex-shrink-0 w-6 h-6 rounded-full bg-blue-50 flex items-center justify-center">
              <span className="text-blue-500 text-[13px]">🧠</span>
            </div>
            <div className="flex-1 text-[15px] text-[#1D1D1F] leading-relaxed">
              <ReactMarkdown>{stepData.think}</ReactMarkdown>
            </div>
          </div>

          {/* 动作 (Action) */}
          <div className="flex gap-4 items-center">
             <div className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-50 flex items-center justify-center">
              <span className="text-purple-500 text-[13px]">🛠️</span>
            </div>
            <code className="flex-1 bg-[#F5F5F7] rounded-lg p-3 text-[14px] text-[#1D1D1F] font-mono border border-gray-100">
              {stepData.action}
            </code>
          </div>

          {/* 观察 (Observation) - macOS 终端风格 */}
          <div className="flex gap-4">
             <div className="mt-1 flex-shrink-0 w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center">
              <span className="text-gray-500 text-[13px]">👁️</span>
            </div>
            <div className="flex-1 bg-[#1E1E1E] rounded-xl overflow-hidden shadow-inner flex flex-col">
              {/* macOS 红黄绿红绿灯顶部栏 */}
              <div className="h-8 bg-[#2D2D2D] flex items-center px-4 gap-2 border-b border-white/10">
                <div className="w-2.5 h-2.5 rounded-full bg-[#FF5F56]"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-[#FFBD2E]"></div>
                <div className="w-2.5 h-2.5 rounded-full bg-[#27C93F]"></div>
                <span className="ml-2 text-[11px] text-gray-400 font-mono">Observation Result</span>
              </div>
              <div className="p-4 overflow-x-auto max-h-72 overflow-y-auto dark-scrollbar">
                <pre className="whitespace-pre-wrap font-mono text-[13px] text-[#E2E2E6] leading-relaxed">
                  {stepData.observation}
                </pre>
              </div>
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

  // 初始化拉取工具列表
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/tools")
      .then(res => res.json())
      .then(data => {
        if (data.tools) {
          setAvailableTools(data.tools);
          setSelectedTools([]);
        }
      })
      .catch(err => {
        console.error("无法获取工具列表:", err);
        setError("无法连接到后端获取工具列表，请检查 FastAPI 是否运行。");
      });
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    if (centerPanelRef.current) {
      centerPanelRef.current.scrollTop = centerPanelRef.current.scrollHeight;
    }
  }, [steps, finalAnswer]);

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

  // Apple 风格的 Markdown 渲染组件
  const markdownComponents = {
    h1: ({node, ...props}: any) => <h1 className="text-2xl font-bold mt-6 mb-3 text-[#1D1D1F]" {...props} />,
    h2: ({node, ...props}: any) => <h2 className="text-xl font-semibold mt-6 mb-3 text-[#1D1D1F] border-b border-gray-100 pb-2" {...props} />,
    h3: ({node, ...props}: any) => <h3 className="text-[17px] font-semibold mt-4 mb-2 text-[#1D1D1F]" {...props} />,
    p: ({node, ...props}: any) => <p className="mb-4 text-[15px] leading-[1.6] text-[#333336]" {...props} />,
    ul: ({node, ...props}: any) => <ul className="list-disc pl-5 mb-4 space-y-2 text-[15px] text-[#333336]" {...props} />,
    ol: ({node, ...props}: any) => <ol className="list-decimal pl-5 mb-4 space-y-2 text-[15px] text-[#333336]" {...props} />,
    li: ({node, ...props}: any) => <li className="leading-[1.6]" {...props} />,
    a: ({node, ...props}: any) => <a className="text-[#007AFF] hover:underline" target="_blank" rel="noreferrer" {...props} />,
    blockquote: ({node, ...props}: any) => <blockquote className="border-l-4 border-[#007AFF] pl-4 italic text-[#86868B] my-5 bg-blue-50/30 py-2 pr-4 rounded-r-lg" {...props} />,
    code: ({node, inline, ...props}: any) => inline
        ? <code className="bg-[#F2F2F7] text-[#D83B01] px-1.5 py-0.5 rounded-md text-[13px] font-mono" {...props} />
        : <code className="block bg-[#F2F2F7] p-3 rounded-xl text-[13px] font-mono overflow-x-auto my-3" {...props} />
  };

  return (
    <div className="flex h-screen bg-[#F5F5F7] text-[#1D1D1F] font-sans overflow-hidden">

      {/* ================= 左侧栏：控制中心 ================= */}
      <div className="w-[340px] bg-white border-r border-gray-200/60 flex flex-col shadow-[2px_0_10px_rgba(0,0,0,0.02)] z-20 shrink-0">
        <div className="p-6 pt-8 pb-4">
          <h1 className="text-[22px] font-bold tracking-tight text-[#1D1D1F] flex items-center gap-2">
            <svg className="w-6 h-6 text-[#007AFF]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
            IterResearch
          </h1>
          <p className="text-[13px] text-[#86868B] mt-1 font-medium">Autonomous Deep Research Agent</p>
        </div>

        <div className="px-6 flex-1 flex flex-col gap-6 overflow-y-auto pb-6 pt-2">
          {/* 输入框区域：无边框感设计 */}
          <div>
            <label className="block text-[13px] font-semibold text-[#86868B] mb-2 uppercase tracking-wider">Research Question</label>
            <div className="relative">
              <textarea
                className="w-full bg-[#F2F2F7] text-[#1D1D1F] p-4 rounded-2xl focus:ring-2 focus:ring-[#007AFF] focus:bg-white outline-none transition-all resize-none h-36 text-[15px] placeholder-[#A1A1A6] shadow-inner"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe what you want to research..."
                disabled={isRunning}
              />
            </div>
          </div>

          {/* 工具选择器区域：iOS 列表交互风格 */}
          <div className="flex-1 flex flex-col min-h-[250px]">
            <label className="block text-[13px] font-semibold text-[#86868B] mb-2 uppercase tracking-wider flex justify-between items-center">
              <span>Enabled Tools</span>
              <span className="font-mono bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">{selectedTools.length}</span>
            </label>

            <div className="space-y-2 flex-1 overflow-y-auto -mx-2 px-2 pb-2">
              {availableTools.length === 0 ? (
                <p className="text-[13px] text-[#86868B] text-center py-6">Loading tools...</p>
              ) : (
                availableTools.map((tool, index) => {
                  const isChecked = selectedTools.includes(index);

                  return (
                    <label
                      key={index}
                      className={`flex items-center gap-4 p-3.5 rounded-2xl cursor-pointer transition-all duration-200 border-2 ${
                        isChecked 
                          ? 'bg-[#E5F1FF] border-[#E5F1FF]' 
                          : 'bg-white border-transparent hover:bg-[#F2F2F7]'
                      }`}
                    >
                      <input
                        type="checkbox"
                        className="sr-only"
                        checked={isChecked}
                        onChange={() => handleToolToggle(index)}
                        disabled={isRunning}
                      />
                      <div className="flex flex-col flex-1">
                        <span className={`text-[14px] font-semibold flex items-center gap-1.5 ${isChecked ? 'text-[#005bb5]' : 'text-[#1D1D1F]'}`}>
                          {tool.type === 'local' ? '📍' : '🌐'} {tool.name}
                        </span>
                        <span className={`text-[12px] mt-0.5 leading-snug line-clamp-2 ${isChecked ? 'text-[#007AFF]/80' : 'text-[#86868B]'}`}>
                          {tool.description}
                        </span>
                      </div>

                      {/* iOS 选中标记 */}
                      <div className={`w-5 h-5 flex-shrink-0 rounded-full flex items-center justify-center transition-opacity ${isChecked ? 'opacity-100 bg-[#007AFF]' : 'opacity-0'}`}>
                         <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>
                      </div>
                    </label>
                  );
                })
              )}
            </div>
          </div>

          {error && (
            <div className="p-3 bg-red-50 text-red-600 rounded-xl text-[13px] font-medium text-center">
              {error}
            </div>
          )}

          {/* 启动按钮：胶囊全圆角 Apple 蓝 */}
          <button
            onClick={startResearch}
            disabled={isRunning || !query.trim() || selectedTools.length === 0}
            className={`w-full py-4 rounded-full font-semibold text-white transition-all duration-200 flex justify-center items-center gap-2 mt-auto text-[15px] ${
              (isRunning || !query.trim() || selectedTools.length === 0)
                ? 'bg-[#E5E5EA] text-[#A1A1A6] cursor-not-allowed' 
                : 'bg-[#007AFF] hover:bg-[#0062CC] active:scale-[0.98] shadow-md shadow-blue-500/20'
            }`}
          >
            {isRunning ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : 'Start Deep Research'}
          </button>
        </div>
      </div>

      {/* ================= 中间区域：执行流 (The Main Stage) ================= */}
      <div className="flex-1 flex flex-col relative bg-[#F5F5F7] min-w-0" ref={centerPanelRef}>

        {/* 悬浮毛玻璃顶部栏 */}
        <div className="sticky top-0 bg-[#F5F5F7]/80 backdrop-blur-md z-10 py-4 px-10 border-b border-gray-200/50">
          <h2 className="text-[15px] font-semibold text-[#1D1D1F]">Execution Trajectory</h2>
        </div>

        <div className="p-10 pb-32 flex flex-col gap-6 overflow-y-auto h-full max-w-4xl mx-auto w-full">
          {steps.map((s, idx) => (
              <StepCard key={idx} stepData={s}/>
          ))}

          {/* 最终报告卡片 */}
          {finalAnswer && (
              <div className="shrink-0 bg-white rounded-2xl shadow-[0_8px_30px_rgba(0,0,0,0.08)] mt-6 overflow-hidden animate-fade-in-up border border-gray-100">
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-100 px-8 py-5 flex items-center gap-3">
                  <span className="text-2xl">🎉</span>
                  <h2 className="text-[18px] font-bold text-[#1D1D1F]">Final Research Report</h2>
                </div>
                <div className="p-8 prose prose-slate max-w-none">
                  <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={markdownComponents}
                  >
                    {finalAnswer}
                  </ReactMarkdown>
                </div>
              </div>
          )}
        </div>
      </div>

      {/* ================= 右侧栏：全局工作区 ================= */}
      <div className="w-[420px] bg-white border-l border-gray-200/60 shadow-[-2px_0_10px_rgba(0,0,0,0.02)] flex flex-col z-20 shrink-0">
        <div className="px-6 py-4 border-b border-gray-100 bg-white flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="font-semibold text-[15px] text-[#1D1D1F]">Global Workspace</h2>
          </div>
          {/* 优雅的呼吸状态指示灯 */}
          <div className="flex items-center gap-2 bg-gray-50 border border-gray-100 px-3 py-1.5 rounded-full">
            <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-[#34C759] animate-pulse' : 'bg-[#D1D1D6]'}`}></div>
            <span className="text-[12px] font-medium text-[#86868B] uppercase tracking-wider">
              {isRunning ? 'Live Updating' : 'Standby'}
            </span>
          </div>
        </div>

        <div className="flex-1 p-8 overflow-y-auto bg-white">
          <div className="max-w-none">
            {liveReport === "Waiting for the agent to start..." ? (
               <div className="flex flex-col items-center justify-center h-full text-[#86868B] mt-32 opacity-60">
                 <svg className="w-16 h-16 mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                 </svg>
                 <p className="text-[14px]">Document will evolve here.</p>
               </div>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
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