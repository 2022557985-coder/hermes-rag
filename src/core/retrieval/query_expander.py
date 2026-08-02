"""Query expansion: synonym expansion, HyDE (Hypothetical Document Embeddings)."""

import json as _json
import logging
import re
from pathlib import Path

logger = logging.getLogger("hermes_rag")


# Chinese synonym dictionary (built-in, extended with ML/technical terms)
# Tuples are immutable → thread-safe
_CN_SYNONYMS: dict[str, tuple[str, ...]] = {
    # IT运维
    "重置": ("初始化", "恢复", "还原", "重设", "复位"),
    "如何": ("怎么", "怎样", "如何做", "怎么做"),
    "密码": ("口令", "密钥", "密码串", "凭据"),
    "设置": ("配置", "设定", "参数", "选项"),
    "安装": ("部署", "配置", "搭建", "设置"),
    "删除": ("移除", "清除", "卸载", "去除"),
    "修改": ("更改", "编辑", "更新", "修正"),
    "查看": ("查询", "浏览", "显示", "展示"),
    "错误": ("故障", "异常", "问题", "报错"),
    "连接": ("链接", "接入", "连通", "联网"),
    "启动": ("开启", "运行", "打开", "激活"),
    "停止": ("关闭", "终止", "暂停", "结束"),
    "配置": ("设置", "参数", "选项", "设定"),
    "文件": ("文档", "档案", "资料", "数据"),
    "系统": ("平台", "框架", "环境", "体系"),
    # ML/技术术语
    "机器学习": ("Machine Learning", "ML", "统计学习", "模式识别"),
    "深度学习": ("Deep Learning", "深度神经网络", "DNN"),
    "神经网络": ("Neural Network", "人工神经网络", "ANN", "深度学习模型"),
    "模型": ("算法", "框架", "架构", "方法"),
    "训练": ("拟合", "学习", "优化", "调参"),
    "预测": ("推理", "推断", "预估", "估算"),
    "分类": ("归类", "识别", "判别", "区分"),
    "回归": ("拟合", "预测", "估计"),
    "聚类": ("分群", "分组", "聚合", "簇分析"),
    "评估": ("评价", "验证", "测试", "衡量"),
    "特征": ("属性", "变量", "维度", "指标"),
    "数据": ("样本", "数据集", "资料", "信息"),
    "算法": ("方法", "模型", "技术", "方案"),
    "性能": ("表现", "效果", "精度", "准确率"),
    "参数": ("超参数", "权重", "系数", "变量"),
    "优化": ("改进", "提升", "增强", "调优"),
    "损失": ("误差", "代价", "目标函数", "Loss"),
    "监督": ("有监督", "有标签", "标注"),
    "交叉验证": ("K折验证", "交叉检验", "CV", "Cross Validation"),
    "向量": ("嵌入", "Embedding", "表征", "表示"),
    "检索": ("搜索", "查询", "召回", "匹配"),
    "生成": ("创建", "产生", "构建", "合成"),
    "Python": ("Python语言", "Python编程", "py"),
    # IT/技术领域扩展
    "数据库": ("Database", "DB", "数据存储", "关系型数据库"),
    "API": ("接口", "应用程序接口", "REST", "端点"),
    "缓存": ("Cache", "缓冲", "临时存储", "高速缓存"),
    "部署": ("上线", "发布", "Deploy", "交付"),
    "监控": ("Monitor", "观测", "追踪", "告警"),
    "日志": ("Log", "记录", "审计", "跟踪"),
    "安全": ("Security", "加密", "防护", "认证"),
    "测试": ("Test", "验证", "检查", "质检"),
    "架构": ("Architecture", "设计", "结构", "框架"),
    "微服务": ("Microservice", "分布式", "服务化", "SOA"),
    "容器": ("Container", "Docker", "虚拟化", "隔离"),
    "Kubernetes": ("K8s", "容器编排", "集群管理"),
    # ML 概念/算法术语 (提升概念查询的稀疏召回)
    "过拟合": ("泛化误差", "训练误差", "欠拟合"),
    "欠拟合": ("过拟合", "泛化能力", "模型复杂度"),
    "随机森林": ("决策树", "集成学习", "投票"),
    "决策树": ("随机森林", "树模型", "特征划分"),
    "逻辑回归": ("Sigmoid", "分类算法", "概率"),
    "支持向量机": ("SVM", "核函数", "超平面"),
    "SVM": ("支持向量机", "核函数", "超平面"),
    "K-means": ("K均值", "聚类", "质心"),
    "K均值": ("K-means", "聚类", "质心"),
    "监督学习": ("有监督", "有标签", "标注"),
    "无监督学习": ("无监督", "无标签", "聚类"),
    "反向传播": ("梯度下降", "前向传播", "损失函数"),
    "激活函数": ("ReLU", "Sigmoid", "Tanh"),
    "卷积": ("卷积核", "特征提取", "滤波器"),
    "池化": ("降采样", "特征图", "最大池化"),
    "ResNet": ("残差网络", "残差连接", "深层网络"),
    "装饰器": ("装饰", "面向切面", "AOP"),
    "编程范式": ("面向对象", "函数式", "过程式"),
    "解释型": ("解释执行", "编译", "运行时"),
    "编译型": ("编译", "解释型", "机器码"),
    "评估指标": ("准确率", "MSE", "MAE"),
    "标准库": ("内置库", "Batteries Included", "第三方库"),
    "函数式编程": ("高阶函数", "lambda", "map"),
    "强化学习": ("环境交互", "奖励", "智能体"),
    "线性回归": ("最小二乘", "残差", "线性关系"),
    "多项式回归": ("高次项", "非线性拟合", "特征扩展"),
    "岭回归": ("L2正则化", "正则化", "Lasso"),
    "Lasso": ("L1正则化", "岭回归", "正则化"),
    "决策树回归": ("回归树", "连续值", "递归划分"),
    "应用领域": ("数据科学", "人工智能", "Web开发"),
    "类型": ("分类", "类别", "种类"),
    "数据科学": ("数据分析", "科学计算", "数据挖掘"),
    "人工智能": ("AI", "机器学习", "深度学习"),
    "适合": ("适用", "擅长", "用于"),
    # 中英跨语言映射（提升跨语言召回）
    "行星": ("planet", "planets", "星球"),
    "太阳系": ("solar system", "行星系统"),
    "山脉": ("mountain range", "mountain", "mountains"),
    "火山": ("volcano", "volcanoes"),
    "芯片": ("chip", "semiconductor", "晶圆"),
    "半导体": ("semiconductor", "chip", "晶圆"),
    "光刻机": ("lithography", "lithography machine", "EUV"),
    "代工": ("foundry", "晶圆代工"),
    "对乙酰氨基酚": ("acetaminophen", "paracetamol", "扑热息痛"),
    "布洛芬": ("ibuprofen", "异丁苯丙酸"),
    "药物": ("drug", "medicine", "medication"),
    "温度": ("temperature", "气温", "升温"),
    "海洋": ("ocean", "sea"),
    "研发": ("R&D", "research and development", "科研"),
    "营收": ("revenue", "收入", "营业额"),
    "市场份额": ("market share", "份额", "占有率"),
    "历史事件": ("historical event", "历史"),
    "官方语言": ("official language"),
    "面积": ("area", "国土面积"),
    "内存安全": ("memory safety", "内存"),
    "编程语言": ("programming language"),
}

# English synonym dictionary
# Tuples are immutable → thread-safe
_EN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "reset": ("initialize", "restore", "revert", "reinitialize"),
    "how": ("how to", "way to", "method for", "approach to"),
    "password": ("passphrase", "credential", "secret", "key"),
    "settings": ("configuration", "options", "preferences", "parameters"),
    "install": ("deploy", "setup", "configure", "set up"),
    "delete": ("remove", "clear", "uninstall", "purge"),
    "modify": ("change", "edit", "update", "alter"),
    "view": ("query", "display", "show", "browse"),
    "error": ("fault", "exception", "bug", "issue", "problem"),
    "connect": ("link", "attach", "join", "pair"),
    "start": ("launch", "begin", "initiate", "open"),
    "stop": ("halt", "terminate", "pause", "end", "quit"),
    "configure": ("setup", "arrange", "tune", "adjust"),
    "file": ("document", "data", "record", "archive"),
    "system": ("platform", "framework", "environment", "infrastructure"),
    # English technical terms
    "machine learning": ("ML", "statistical learning", "pattern recognition", "AI"),
    "deep learning": ("DL", "DNN", "neural network", "deep neural network"),
    "API": ("interface", "endpoint", "service", "REST API"),
    "database": ("DB", "data store", "storage", "RDBMS"),
    "cache": ("buffer", "temporary storage", "memoization"),
    "deployment": ("release", "rollout", "delivery", "launch"),
    "monitoring": ("observability", "tracking", "alerting", "logging"),
    "security": ("encryption", "protection", "authentication", "authorization"),
    "svm": ("support vector machine", "kernel", "hyperplane"),
    "k-means": ("kmeans", "clustering", "centroid"),
    "cnn": ("convolutional neural network", "convolution", "pooling"),
    "resnet": ("residual network", "residual connection", "deep network"),
    # 中英跨语言映射（提升跨语言召回）
    "planet": ("行星", "planets", "星球"),
    "solar system": ("太阳系", "行星系统"),
    "mountain range": ("山脉", "山系"),
    "volcano": ("火山", "休眠火山"),
    "memory safety": ("内存安全", "内存"),
    "programming language": ("编程语言", "程序语言"),
    "chip": ("芯片", "晶圆", "半导体"),
    "semiconductor": ("半导体", "芯片"),
    "lithography": ("光刻", "光刻机"),
    "foundry": ("代工厂", "晶圆代工", "台积电"),
    "acetaminophen": ("对乙酰氨基酚", "扑热息痛"),
    "paracetamol": ("对乙酰氨基酚", "扑热息痛"),
    "ibuprofen": ("布洛芬", "异丁苯丙酸"),
    "surface temperature": ("表面温度", "地表温度"),
    "market share": ("市场份额", "市场占有率"),
    "official language": ("官方语言"),
    "revenue": ("营收", "收入", "营业额"),
    "research and development": ("研发", "R&D"),
    "temperature": ("温度", "气温"),
    "ocean": ("海洋"),
    "historical event": ("历史事件"),
}

# Stop words for keyword extraction — frozenset is immutable and thread-safe
_STOP_WORDS: frozenset = frozenset({
    # Chinese stop words
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "为什么", "哪", "吗", "吧", "呢", "啊", "哦",
    "与", "或", "但", "而", "且", "从", "把", "被", "让", "给", "对",
    "以", "及", "向", "所", "其", "为", "之", "将", "已", "可", "能",
    "如果", "因为", "所以", "虽然", "但是", "然而", "然后", "接着",
    "目前", "现在", "已经", "正在", "可以", "需要", "应该", "能够",
    "使用", "进行", "通过", "根据", "关于", "对于", "为了", "按照",
    "这个", "那个", "哪个", "这些", "那些", "这里", "那里", "哪里",
    # English stop words
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "and", "but", "or", "not", "no", "nor", "so", "if",
    "than", "too", "very", "just", "then", "now", "also", "about",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "here", "there",
})

# Pre-compiled regex for mixed Chinese-English tokenization
_TOKENIZE_RE: re.Pattern = re.compile(
    r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+|[^\s]"
)


# Extra synonyms loaded from a JSON file so additions are hot-reloadable
# without restarting the server. {"cn": {...}, "en": {...}}
_EXTRA_SYNONYMS_PATH = Path(__file__).parent / "extra_synonyms.json"


def _load_extra_synonyms() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Load extra synonym maps from JSON (hot-reloadable), else empty dicts."""
    try:
        if _EXTRA_SYNONYMS_PATH.exists():
            data = _json.loads(_EXTRA_SYNONYMS_PATH.read_text(encoding="utf-8-sig"))
            cn = {k: tuple(v) for k, v in data.get("cn", {}).items()}
            en = {k: tuple(v) for k, v in data.get("en", {}).items()}
            return cn, en
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return {}, {}


class QueryExpander:
    """Expand queries with synonyms and optional HyDE."""

    def __init__(
        self,
        synonym_enabled: bool = True,
        hyde_enabled: bool = False,
        hyde_model: str = "google-t5/t5-small",
        max_synonyms: int = 3,
    ):
        self.synonym_enabled: bool = synonym_enabled
        self.hyde_enabled: bool = hyde_enabled
        self.hyde_model: str = hyde_model
        self.max_synonyms: int = max_synonyms
        self._hyde_model = None
        self._hyde_tokenizer = None

    def expand(self, query: str) -> dict:
        """Expand a query with synonyms and optional HyDE.

        Args:
            query: Original query string.

        Returns:
            dict with:
                - original: Original query
                - expanded: Expanded query string
                - hyde_text: HyDE generated text (None if disabled)
                - synonyms: List of synonym expansions
                - weighted_synonyms: List of (synonym, weight) tuples
                - keywords: List of extracted key terms
        """
        # Input validation: reject empty or whitespace-only queries
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Query expansion received empty or invalid query")
            return {
                "original": query if isinstance(query, str) else "",
                "expanded": query if isinstance(query, str) else "",
                "hyde_text": None,
                "synonyms": [],
                "weighted_synonyms": [],
                "keywords": [],
            }

        result: dict[str, object] = {
            "original": query,
            "expanded": query,
            "hyde_text": None,
            "synonyms": [],
            "weighted_synonyms": [],
            "keywords": [],
        }

        # Tokenize the query for downstream processing
        tokens: list[str] = self._tokenize_query(query)

        if self.synonym_enabled:
            # Weighted synonyms (exact match > partial match)
            weighted_syns: list[tuple[str, float]] = self._get_synonyms_with_weights(
                query, tokens
            )
            result["weighted_synonyms"] = weighted_syns

            # Backward-compatible flat synonym list
            synonyms: list[str] = self._get_synonyms(query)
            result["synonyms"] = synonyms

            # Extract key terms as boost keywords
            keywords: list[str] = self._expand_with_keywords(query, tokens)
            result["keywords"] = keywords

            # Build expanded query: original + synonyms + boost keywords
            expansion_parts: list[str] = []
            if synonyms:
                expansion_parts.extend(synonyms)
            # Only add boost keywords when there are actual synonyms found.
            # Standalone keywords without synonyms just repeat query terms
            # and introduce noise.
            if synonyms:
                synonym_set: set[str] = set(s.lower() for s in synonyms)
                query_lower: str = query.lower()
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in synonym_set and kw_lower != query_lower and kw_lower not in query_lower:
                        expansion_parts.append(kw)

            if expansion_parts:
                result["expanded"] = query + " " + " ".join(expansion_parts)

        if self.hyde_enabled:
            hyde_text: str | None = self._generate_hyde(query)
            if hyde_text:
                result["hyde_text"] = hyde_text

        return result

    def _tokenize_query(self, query: str) -> list[str]:
        """Tokenize a mixed Chinese-English query into individual tokens.

        Uses regex to split on:
        - Chinese character sequences (\\u4e00-\\u9fff)
        - English word sequences ([a-zA-Z]+)
        - Digit sequences (\\d+)
        - Other non-whitespace characters

        Args:
            query: The raw query string.

        Returns:
            List of token strings.
        """
        return _TOKENIZE_RE.findall(query)

    def _get_synonyms(self, query: str) -> list[str]:
        """Get synonyms for words in the query (supports both Chinese and English)."""
        synonyms: list[str] = []
        extra_cn, extra_en = _load_extra_synonyms()
        cn_map = {**_CN_SYNONYMS, **extra_cn}
        en_map = {**_EN_SYNONYMS, **extra_en}
        # Search Chinese synonyms
        for word, syns in cn_map.items():
            if word in query:
                for s in syns[: self.max_synonyms]:
                    if s not in query:
                        synonyms.append(s)
        # Search English synonyms (case-insensitive)
        query_lower: str = query.lower()
        for word, syns in en_map.items():
            if word in query_lower:
                for s in syns[: self.max_synonyms]:
                    if s.lower() not in query_lower:
                        synonyms.append(s)
        return synonyms

    def _get_synonyms_with_weights(
        self, query: str, tokens: list[str]
    ) -> list[tuple[str, float]]:
        """Get synonyms with weights based on match quality.

        Exact match (token == dict key) → weight 1.0
        Partial match (token in key or key in token) → weight 0.5

        Args:
            query: The original query string for deduplication.
            tokens: Pre-tokenized query tokens.

        Returns:
            List of (synonym, weight) tuples, sorted by weight descending.
        """
        weighted: list[tuple[str, float]] = []
        query_lower: str = query.lower()
        seen: set[str] = set()
        extra_cn, extra_en = _load_extra_synonyms()
        cn_map = {**_CN_SYNONYMS, **extra_cn}
        en_map = {**_EN_SYNONYMS, **extra_en}

        for token in tokens:
            token_lower: str = token.lower()

            # Check Chinese synonyms
            for word, syns in cn_map.items():
                if token == word:
                    # Exact match
                    for s in syns[: self.max_synonyms]:
                        if s.lower() not in query_lower and s.lower() not in seen:
                            weighted.append((s, 1.0))
                            seen.add(s.lower())
                elif token in word or word in token:
                    # Partial match (only if no exact match for this token-word pair)
                    for s in syns[: self.max_synonyms]:
                        if s.lower() not in query_lower and s.lower() not in seen:
                            weighted.append((s, 0.5))
                            seen.add(s.lower())

            # Check English synonyms (case-insensitive)
            for word, syns in en_map.items():
                word_lower: str = word.lower()
                if token_lower == word_lower:
                    for s in syns[: self.max_synonyms]:
                        if s.lower() not in query_lower and s.lower() not in seen:
                            weighted.append((s, 1.0))
                            seen.add(s.lower())
                elif token_lower in word_lower or word_lower in token_lower:
                    for s in syns[: self.max_synonyms]:
                        if s.lower() not in query_lower and s.lower() not in seen:
                            weighted.append((s, 0.5))
                            seen.add(s.lower())

        # Sort by weight descending, then by synonym alphabetically for stability
        weighted.sort(key=lambda x: (-x[1], x[0]))
        return weighted

    def _expand_with_keywords(
        self, query: str, tokens: list[str]
    ) -> list[str]:
        """Extract key terms from the query to use as boost terms.

        Filters out stop words, short tokens, and tokens already in the query.
        Useful for adding domain-specific boost terms to the expanded query.

        Args:
            query: The original query string.
            tokens: Pre-tokenized query tokens.

        Returns:
            List of extracted keyword strings.
        """
        keywords: list[str] = []
        query.lower()
        seen: set[str] = set()

        for token in tokens:
            token_lower: str = token.lower()

            # Skip stop words
            if token_lower in _STOP_WORDS:
                continue

            # Skip tokens already present in the query (case-insensitive)
            if token_lower in seen:
                continue

            # Skip very short tokens (single ASCII char, single CJK char)
            if len(token) <= 1:
                continue

            # Skip pure numeric tokens
            if token.isdigit():
                continue

            # Token is a meaningful keyword
            keywords.append(token)
            seen.add(token_lower)

        return keywords

    def _generate_hyde(self, query: str) -> str | None:
        """Generate a hypothetical document using T5-small.

        This is disabled by default due to noise concerns.
        """
        if not self.hyde_enabled:
            return None

        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            if self._hyde_model is None:
                self._hyde_tokenizer = T5Tokenizer.from_pretrained(
                    self.hyde_model, local_files_only=True
                )
                self._hyde_model = T5ForConditionalGeneration.from_pretrained(
                    self.hyde_model, local_files_only=True
                )

            input_text: str = f"Please write a passage to answer the question: {query}"
            inputs = self._hyde_tokenizer(
                input_text, return_tensors="pt", max_length=128, truncation=True
            )
            outputs = self._hyde_model.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            hyde_text: str = self._hyde_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return hyde_text
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"HyDE generation failed: {e}")
            return None