using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

internal static class Program
{
    private static readonly ToolRegistry Registry = BuildToolRegistry();
    private static readonly OpenAiClient OpenAi = new OpenAiClient();
    private static readonly LLMPlanner PlannerV2 = new LLMPlanner(OpenAi);
    private static readonly MemoryManager Memory = BuildMemoryManager();

    private static void Main()
    {
        PrintWelcome();
        RunAgentLoop();
    }

    private static ToolRegistry BuildToolRegistry()
    {
        ToolRegistry registry = new ToolRegistry();
        registry.Register(new WeatherTool());
        registry.Register(new SummaryTool());
        return registry;
    }

    private static MemoryManager BuildMemoryManager()
    {
        return new MemoryManager(new ShortTermMemory(14), new LongTermMemory(), new Retriever(), new MemoryCompressionUnit(), 10, 2000);
    }

    private static void RunAgentLoop()
    {
        while (true)
        {
            string? input = ReadUserInput();
            if (input is null)
            {
                RenderOutput("系统", "检测到输入结束，程序退出。");
                return;
            }

            if (string.IsNullOrWhiteSpace(input))
            {
                RenderOutput("系统", "请输入内容。输入 /help 查看说明。");
                continue;
            }

            IntentType intent = RouteIntent(input);
            HandlerResult result = DispatchToHandler(intent, input);
            RenderOutput(result.Channel, result.Message);
            Memory.ProcessTurn(input, result.Message, intent);

            if (result.ShouldExit)
            {
                return;
            }
        }
    }

    private static string? ReadUserInput()
    {
        Console.Write("[You] > ");
        return Console.ReadLine();
    }

    private static IntentType RouteIntent(string input)
    {
        string trimmed = input.Trim();
        if (trimmed.StartsWith("/", StringComparison.Ordinal)) return IntentType.Command;
        return ContainsTaskKeywords(trimmed) ? IntentType.Task : IntentType.Chat;
    }

    private static bool ContainsTaskKeywords(string input)
    {
        return input.Contains("帮我", StringComparison.Ordinal)
            || input.Contains("请", StringComparison.Ordinal)
            || input.Contains("执行", StringComparison.Ordinal)
            || input.Contains("生成", StringComparison.Ordinal)
            || input.Contains("查", StringComparison.Ordinal);
    }

    private static HandlerResult DispatchToHandler(IntentType intent, string input)
    {
        return intent switch
        {
            IntentType.Chat => ChatHandler(input),
            IntentType.Task => TaskHandler(input),
            IntentType.Command => CommandHandler(input),
            _ => new HandlerResult("系统", "未知意图。", false)
        };
    }

    private static HandlerResult ChatHandler(string input)
    {
        List<string> ctx = Memory.Retrieve(input, 3);
        string systemPrompt = "你是 CLI Agent 的聊天模块，回答简洁、友好。";
        if (ctx.Count > 0)
        {
            systemPrompt += " 可参考记忆上下文: " + string.Join(" | ", ctx);
        }

        if (OpenAi.TryChat(systemPrompt, input, out string? answer) && !string.IsNullOrWhiteSpace(answer))
        {
            return new HandlerResult("ChatHandler", answer.Trim(), false);
        }

        return new HandlerResult("ChatHandler", $"收到聊天消息：\"{input.Trim()}\"。", false);
    }

    private static HandlerResult TaskHandler(string input)
    {
        string task = NormalizeTaskText(input);
        List<string> context = Memory.Retrieve(task, 5);

        Plan plan;
        if (!PlannerV2.TryCreatePlan(task, context, out plan))
        {
            plan = LegacyPlanner(task);
        }

        List<string> lines = Executor(plan, Registry);
        string message = lines.Count == 0 ? "未识别到可执行步骤。" : string.Join(Environment.NewLine, lines);
        return new HandlerResult("TaskHandler", message, false);
    }

    private static string NormalizeTaskText(string input)
    {
        string trimmed = input.Trim();
        string[] prefixes = { "帮我", "请", "请帮我" };
        foreach (string prefix in prefixes)
        {
            if (trimmed.StartsWith(prefix, StringComparison.Ordinal)) return trimmed.Substring(prefix.Length).Trim();
        }

        return trimmed;
    }

    private static Plan LegacyPlanner(string taskInput)
    {
        string[] segments = Regex.Split(taskInput, "并|然后|再", RegexOptions.None)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToArray();

        List<PlanStep> steps = new List<PlanStep>();
        for (int i = 0; i < segments.Length; i++)
        {
            string tool = LegacyToolRouter(segments[i]);
            Dictionary<string, string> parameters = LegacyParameterExtractor(segments[i], i + 1);
            steps.Add(new PlanStep(i + 1, segments[i], tool, parameters));
        }

        return new Plan(steps);
    }

    private static string LegacyToolRouter(string description)
    {
        if (description.Contains("天气", StringComparison.Ordinal)) return "weather";
        if (description.Contains("总结", StringComparison.Ordinal)) return "summary";
        return string.Empty;
    }

    private static Dictionary<string, string> LegacyParameterExtractor(string description, int stepId)
    {
        Dictionary<string, string> parameters = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (description.Contains("北京", StringComparison.Ordinal)) parameters["city"] = "北京";
        else if (description.Contains("上海", StringComparison.Ordinal)) parameters["city"] = "上海";

        if (description.Contains("明天", StringComparison.Ordinal)) parameters["date"] = "明天";
        else if (description.Contains("今天", StringComparison.Ordinal)) parameters["date"] = "今天";

        if (description.Contains("总结", StringComparison.Ordinal)) parameters["text"] = stepId > 1 ? $"$step{stepId - 1}" : description;
        return parameters;
    }

    private static List<string> Executor(Plan plan, ToolRegistry registry)
    {
        Dictionary<int, string> context = new Dictionary<int, string>();
        List<string> outputs = new List<string>();

        foreach (PlanStep step in plan.Steps.OrderBy(s => s.Id))
        {
            string line = ExecuteStep(step, registry, context);
            outputs.Add(line);
            context[step.Id] = ExtractResultText(line);
        }

        return outputs;
    }

    private static string ExecuteStep(PlanStep step, ToolRegistry registry, Dictionary<int, string> context)
    {
        ITool? tool = registry.Get(step.Tool);
        if (tool is null) return $"[Step {step.Id}] Executed: {step.Description}";

        Dictionary<string, string> resolvedParameters = ResolveStepReferences(step.Parameters, context);
        return $"[Step {step.Id}] {tool.Execute(resolvedParameters)}";
    }

    private static Dictionary<string, string> ResolveStepReferences(Dictionary<string, string> parameters, Dictionary<int, string> context)
    {
        Dictionary<string, string> resolved = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (KeyValuePair<string, string> pair in parameters)
        {
            if (pair.Value.StartsWith("$step", StringComparison.OrdinalIgnoreCase)
                && int.TryParse(pair.Value.Substring(5), out int refStep)
                && context.TryGetValue(refStep, out string? refResult))
            {
                resolved[pair.Key] = refResult;
            }
            else
            {
                resolved[pair.Key] = pair.Value;
            }
        }

        return resolved;
    }

    private static string ExtractResultText(string executionLine)
    {
        int idx = executionLine.IndexOf("] ", StringComparison.Ordinal);
        return idx >= 0 && idx + 2 < executionLine.Length ? executionLine[(idx + 2)..] : executionLine;
    }

    private static HandlerResult CommandHandler(string input)
    {
        string cmd = input.Trim();
        if (cmd.Equals("/exit", StringComparison.OrdinalIgnoreCase)) return new HandlerResult("CommandHandler", "收到 /exit，程序退出。", true);
        if (cmd.Equals("/help", StringComparison.OrdinalIgnoreCase)) return new HandlerResult("CommandHandler", BuildHelpText(), false);
        if (cmd.Equals("/memory", StringComparison.OrdinalIgnoreCase)) return new HandlerResult("CommandHandler", Memory.GetMemorySnapshot(), false);
        return new HandlerResult("CommandHandler", $"未知命令：{cmd}。", false);
    }

    private static void RenderOutput(string channel, string message)
    {
        Console.WriteLine($"[{channel}] {message}\n");
    }

    private static void PrintWelcome()
    {
        Console.WriteLine("=== Agent CLI (OpenAI Chat + Planner) ===");
        Console.WriteLine("输入 /help 查看说明，输入 /exit 退出。\n");
        Console.WriteLine("提示：设置 OPENAI_API_KEY 后，Chat 和 Planner 将调用 OpenAI API。\n");
    }

    private static string BuildHelpText()
    {
        return string.Join(Environment.NewLine, new[]
        {
            "可用命令：/help, /exit, /memory",
            "OPENAI_API_KEY: OpenAI API 密钥（可选，未设置时自动回退规则逻辑）",
            "OPENAI_MODEL: 可选模型名，默认 gpt-4.1-mini"
        });
    }
}

internal enum IntentType { Chat, Task, Command }

internal interface ITool
{
    string Name { get; }
    string Description { get; }
    IReadOnlyList<ToolParameter> Parameters { get; }
    string Execute(Dictionary<string, string> parameters);
}

internal sealed class ToolParameter
{
    public ToolParameter(string name, string description, bool required) { Name = name; Description = description; Required = required; }
    public string Name { get; }
    public string Description { get; }
    public bool Required { get; }
}

internal sealed class WeatherTool : ITool
{
    public string Name => "weather";
    public string Description => "返回模拟天气";
    public IReadOnlyList<ToolParameter> Parameters => new List<ToolParameter> { new ToolParameter("city", "城市", false), new ToolParameter("date", "日期", false) };

    public string Execute(Dictionary<string, string> parameters)
    {
        string city = parameters.TryGetValue("city", out string? c) ? c : "默认城市";
        string date = parameters.TryGetValue("date", out string? d) ? d : "今天";
        return $"{city} {date} 天气：晴 25°C";
    }
}

internal sealed class SummaryTool : ITool
{
    public string Name => "summary";
    public string Description => "返回简单总结";
    public IReadOnlyList<ToolParameter> Parameters => new List<ToolParameter> { new ToolParameter("text", "总结文本", true) };
    public string Execute(Dictionary<string, string> parameters)
    {
        string text = parameters.TryGetValue("text", out string? t) ? t : "无可总结内容";
        return $"总结：{text}";
    }
}

internal sealed class ToolRegistry
{
    private readonly Dictionary<string, ITool> _tools = new Dictionary<string, ITool>(StringComparer.OrdinalIgnoreCase);
    public void Register(ITool tool) => _tools[tool.Name] = tool;
    public ITool? Get(string toolName) => _tools.TryGetValue(toolName, out ITool? tool) ? tool : null;
}

internal sealed class LLMPlanner
{
    private readonly OpenAiClient _openAi;

    public LLMPlanner(OpenAiClient openAi)
    {
        _openAi = openAi;
    }

    public bool TryCreatePlan(string userTask, List<string> memoryContext, out Plan plan)
    {
        plan = new Plan(new List<PlanStep>());

        string prompt = BuildPrompt(userTask, memoryContext);
        if (_openAi.TryJson(prompt, out string? jsonFromApi) && TryParsePlan(jsonFromApi!, out plan))
        {
            return true;
        }

        string mock = CallMockLlm(userTask);
        return TryParsePlan(mock, out plan);
    }

    private static string BuildPrompt(string userTask, List<string> memoryContext)
    {
        string ctx = memoryContext.Count == 0 ? "(empty)" : string.Join(" | ", memoryContext);
        return string.Join("\n", new[]
        {
            "你是任务规划器。",
            "把用户任务转换为 JSON Plan。",
            "只输出 JSON，不允许解释。",
            "JSON 必须包含 steps。",
            "每个 step 必须包含 id, description, tool, parameters。",
            "schema: {\"steps\":[{\"id\":1,\"description\":\"...\",\"tool\":\"weather|summary\",\"parameters\":{}}]}",
            $"memory_context: {ctx}",
            $"task: {userTask}"
        });
    }

    private static string CallMockLlm(string userTask)
    {
        if (userTask.Contains("天气", StringComparison.Ordinal) && userTask.Contains("总结", StringComparison.Ordinal))
        {
            string city = userTask.Contains("上海", StringComparison.Ordinal) ? "上海" : "北京";
            return JsonSerializer.Serialize(new
            {
                steps = new object[]
                {
                    new { id = 1, description = $"查{city}天气", tool = "weather", parameters = new Dictionary<string, string> { ["city"] = city, ["date"] = "今天" } },
                    new { id = 2, description = "总结天气", tool = "summary", parameters = new Dictionary<string, string> { ["text"] = "$step1" } }
                }
            });
        }

        return JsonSerializer.Serialize(new
        {
            steps = new object[]
            {
                new { id = 1, description = userTask, tool = "", parameters = new Dictionary<string, string>() }
            }
        });
    }

    private static bool TryParsePlan(string json, out Plan plan)
    {
        plan = new Plan(new List<PlanStep>());
        try
        {
            using JsonDocument doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("steps", out JsonElement stepsEl) || stepsEl.ValueKind != JsonValueKind.Array)
            {
                return false;
            }

            List<PlanStep> steps = new List<PlanStep>();
            foreach (JsonElement item in stepsEl.EnumerateArray())
            {
                int id = item.TryGetProperty("id", out JsonElement idEl) && idEl.TryGetInt32(out int parsedId) ? parsedId : 0;
                string desc = item.TryGetProperty("description", out JsonElement dEl) ? dEl.GetString() ?? string.Empty : string.Empty;
                string tool = item.TryGetProperty("tool", out JsonElement tEl) ? tEl.GetString() ?? string.Empty : string.Empty;
                Dictionary<string, string> parameters = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

                if (item.TryGetProperty("parameters", out JsonElement pEl) && pEl.ValueKind == JsonValueKind.Object)
                {
                    foreach (JsonProperty p in pEl.EnumerateObject())
                    {
                        parameters[p.Name] = p.Value.ValueKind == JsonValueKind.String ? p.Value.GetString() ?? string.Empty : p.Value.ToString();
                    }
                }

                if (id > 0 && !string.IsNullOrWhiteSpace(desc)) steps.Add(new PlanStep(id, desc, tool, parameters));
            }

            plan = new Plan(steps);
            return true;
        }
        catch
        {
            return false;
        }
    }
}

internal sealed class OpenAiClient
{
    private readonly string? _apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
    private readonly string _model = Environment.GetEnvironmentVariable("OPENAI_MODEL") ?? "gpt-4.1-mini";
    private static readonly HttpClient Http = new HttpClient();

    public bool TryChat(string systemPrompt, string userInput, out string? answer)
    {
        answer = null;
        if (string.IsNullOrWhiteSpace(_apiKey)) return false;

        string prompt = systemPrompt + "\n用户输入: " + userInput;
        return TryJson(prompt, out answer);
    }

    public bool TryJson(string prompt, out string? output)
    {
        output = null;
        if (string.IsNullOrWhiteSpace(_apiKey)) return false;

        try
        {
            HttpRequestMessage req = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/responses");
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

            string payload = JsonSerializer.Serialize(new
            {
                model = _model,
                input = prompt
            });

            req.Content = new StringContent(payload, Encoding.UTF8, "application/json");
            HttpResponseMessage resp = Http.Send(req);
            if (!resp.IsSuccessStatusCode) return false;

            string body = resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            output = ExtractOutputText(body);
            return !string.IsNullOrWhiteSpace(output);
        }
        catch
        {
            return false;
        }
    }

    private static string? ExtractOutputText(string body)
    {
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement root = doc.RootElement;

        if (root.TryGetProperty("output_text", out JsonElement outText) && outText.ValueKind == JsonValueKind.String)
        {
            return outText.GetString();
        }

        if (root.TryGetProperty("output", out JsonElement outputArr) && outputArr.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in outputArr.EnumerateArray())
            {
                if (!item.TryGetProperty("content", out JsonElement content) || content.ValueKind != JsonValueKind.Array) continue;
                foreach (JsonElement c in content.EnumerateArray())
                {
                    if (c.TryGetProperty("text", out JsonElement textEl) && textEl.ValueKind == JsonValueKind.String)
                    {
                        return textEl.GetString();
                    }
                }
            }
        }

        return null;
    }
}

internal sealed class MemoryManager
{
    private readonly ShortTermMemory _stm;
    private readonly LongTermMemory _ltm;
    private readonly Retriever _retriever;
    private readonly MemoryCompressionUnit _compressor;
    private readonly int _compressionTurnThreshold;
    private readonly int _compressionTokenThreshold;
    private int _turnCount;
    private IntentType? _lastIntent;

    public MemoryManager(ShortTermMemory stm, LongTermMemory ltm, Retriever retriever, MemoryCompressionUnit compressor, int compressionTurnThreshold, int compressionTokenThreshold)
    {
        _stm = stm;
        _ltm = ltm;
        _retriever = retriever;
        _compressor = compressor;
        _compressionTurnThreshold = compressionTurnThreshold;
        _compressionTokenThreshold = compressionTokenThreshold;
    }

    public void ProcessTurn(string userInput, string agentOutput, IntentType intent)
    {
        _turnCount++;
        _stm.Append($"USER: {userInput}");
        _stm.Append($"AGENT: {agentOutput}");

        if (ShouldPersistToLongTerm(userInput, agentOutput, intent))
        {
            _ltm.AddRecord(new MemoryRecord(Guid.NewGuid().ToString("N"), DecideMemoryType(userInput, intent), $"USER={userInput} | AGENT={agentOutput}", DateTime.UtcNow));
        }

        if (ShouldCompress(intent)) CompressAndStore();
        _lastIntent = intent;
    }

    public List<string> Retrieve(string query, int topK) => _retriever.Retrieve(_ltm, query, topK);

    public string GetMemorySnapshot()
    {
        List<string> stm = _stm.GetContext();
        List<string> ltm = _retriever.Retrieve(_ltm, string.Empty, 8);
        return "STM:\n" + string.Join("\n", stm) + "\n\nLTM(top):\n" + string.Join("\n", ltm);
    }

    private bool ShouldCompress(IntentType currentIntent)
    {
        bool turnTrigger = _turnCount >= _compressionTurnThreshold;
        bool tokenTrigger = _stm.EstimateTokenSize() >= _compressionTokenThreshold;
        bool stageSwitchTrigger = _lastIntent == IntentType.Task && currentIntent == IntentType.Chat;
        return turnTrigger || tokenTrigger || stageSwitchTrigger;
    }

    private void CompressAndStore()
    {
        List<string> recent = _stm.GetContext();
        if (recent.Count == 0) return;

        MemoryCompressionResult result = _compressor.Compress(recent);
        if (!string.IsNullOrWhiteSpace(result.CompressedSummary))
        {
            _ltm.AddRecord(new MemoryRecord(Guid.NewGuid().ToString("N"), "episodic", result.CompressedSummary, DateTime.UtcNow));
        }

        foreach (string fact in result.ImportantFacts)
        {
            if (!string.IsNullOrWhiteSpace(fact) && !_ltm.ExistsContent(fact))
            {
                _ltm.AddRecord(new MemoryRecord(Guid.NewGuid().ToString("N"), "semantic", fact, DateTime.UtcNow));
            }
        }

        _stm.Clear();
        _stm.Append($"[CompressedSummary] {result.CompressedSummary}");
        _turnCount = 0;
    }

    private static bool ShouldPersistToLongTerm(string userInput, string agentOutput, IntentType intent)
    {
        if (intent == IntentType.Task) return true;
        return userInput.Contains("我喜欢", StringComparison.Ordinal)
            || userInput.Contains("我的", StringComparison.Ordinal)
            || agentOutput.Contains("完成", StringComparison.Ordinal);
    }

    private static string DecideMemoryType(string userInput, IntentType intent)
    {
        if (intent == IntentType.Task) return "episodic";
        if (userInput.Contains("我喜欢", StringComparison.Ordinal)) return "semantic";
        return "procedural";
    }
}

internal sealed class MemoryCompressionUnit
{
    public MemoryCompressionResult Compress(List<string> recentContext)
    {
        List<string> dedup = recentContext.Distinct(StringComparer.Ordinal).ToList();
        string summary = string.Join(" ; ", dedup.Take(4));
        List<string> facts = dedup.Where(x => x.Contains("喜欢", StringComparison.Ordinal) || x.Contains("天气", StringComparison.Ordinal) || x.Contains("总结", StringComparison.Ordinal)).Take(5).ToList();
        return new MemoryCompressionResult(summary, facts);
    }
}

internal readonly record struct MemoryCompressionResult(string CompressedSummary, List<string> ImportantFacts);

internal sealed class ShortTermMemory
{
    private readonly List<string> _buffer = new List<string>();
    private readonly int _capacity;
    public ShortTermMemory(int capacity) { _capacity = Math.Max(1, capacity); }
    public void Append(string text) { _buffer.Add(text); while (_buffer.Count > _capacity) _buffer.RemoveAt(0); }
    public List<string> GetContext() => new List<string>(_buffer);
    public void Clear() => _buffer.Clear();
    public int EstimateTokenSize() => Math.Max(1, _buffer.Sum(x => x.Length) / 4);
}

internal readonly record struct MemoryRecord(string Id, string Type, string Content, DateTime Timestamp);

internal sealed class LongTermMemory
{
    public List<MemoryRecord> SemanticStore { get; } = new List<MemoryRecord>();
    public List<MemoryRecord> EpisodicStore { get; } = new List<MemoryRecord>();
    public List<MemoryRecord> ProceduralStore { get; } = new List<MemoryRecord>();

    public void AddRecord(MemoryRecord record)
    {
        switch (record.Type)
        {
            case "semantic": SemanticStore.Add(record); break;
            case "episodic": EpisodicStore.Add(record); break;
            case "procedural": ProceduralStore.Add(record); break;
            default: EpisodicStore.Add(record); break;
        }
    }

    public bool ExistsContent(string content) => AllRecords().Any(r => r.Content.Equals(content, StringComparison.Ordinal));
    public List<MemoryRecord> AllRecords() => SemanticStore.Concat(EpisodicStore).Concat(ProceduralStore).ToList();
}

internal sealed class Retriever
{
    public List<string> Retrieve(LongTermMemory ltm, string query, int topK)
    {
        IEnumerable<MemoryRecord> records = ltm.AllRecords();
        IEnumerable<string> keywords = SplitKeywords(query);
        if (keywords.Any()) records = records.Where(r => keywords.Any(k => r.Content.Contains(k, StringComparison.OrdinalIgnoreCase)));
        return records.OrderByDescending(r => r.Timestamp).Take(Math.Max(1, topK)).Select(r => r.Content).ToList();
    }

    private static IEnumerable<string> SplitKeywords(string query)
    {
        if (string.IsNullOrWhiteSpace(query)) return Array.Empty<string>();
        return query.Split(new[] { ' ', ',', '，', '.', '。', ';', '；' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .Distinct(StringComparer.OrdinalIgnoreCase);
    }
}

internal readonly record struct PlanStep(int Id, string Description, string Tool, Dictionary<string, string> Parameters);
internal readonly record struct Plan(List<PlanStep> Steps);
internal readonly record struct HandlerResult(string Channel, string Message, bool ShouldExit);
