"""Query expansion: synonym expansion, HyDE (Hypothetical Document Embeddings)."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger("hermes_rag")


# Chinese synonym dictionary (built-in, extended with ML/technical terms)
_CN_SYNONYMS = {
    # IT运维
    "重置": ["初始化", "恢复", "还原", "重设", "复位"],
    "如何": ["怎么", "怎样", "如何做", "怎么做"],
    "密码": ["口令", "密钥", "密码串", "凭据"],
    "设置": ["配置", "设定", "参数", "选项"],
    "安装": ["部署", "配置", "搭建", "设置"],
    "删除": ["移除", "清除", "卸载", "去除"],
    "修改": ["更改", "编辑", "更新", "修正"],
    "查看": ["查询", "浏览", "显示", "展示"],
    "错误": ["故障", "异常", "问题", "报错"],
    "连接": ["链接", "接入", "连通", "联网"],
    "启动": ["开启", "运行", "打开", "激活"],
    "停止": ["关闭", "终止", "暂停", "结束"],
    "配置": ["设置", "参数", "选项", "设定"],
    "文件": ["文档", "档案", "资料", "数据"],
    "系统": ["平台", "框架", "环境", "体系"],
    # ML/技术术语
    "机器学习": ["Machine Learning", "ML", "统计学习", "模式识别"],
    "深度学习": ["Deep Learning", "深度神经网络", "DNN"],
    "神经网络": ["Neural Network", "人工神经网络", "ANN", "深度学习模型"],
    "模型": ["算法", "框架", "架构", "方法"],
    "训练": ["拟合", "学习", "优化", "调参"],
    "预测": ["推理", "推断", "预估", "估算"],
    "分类": ["归类", "识别", "判别", "区分"],
    "回归": ["拟合", "预测", "估计"],
    "聚类": ["分群", "分组", "聚合", "簇分析"],
    "评估": ["评价", "验证", "测试", "衡量"],
    "特征": ["属性", "变量", "维度", "指标"],
    "数据": ["样本", "数据集", "资料", "信息"],
    "算法": ["方法", "模型", "技术", "方案"],
    "性能": ["表现", "效果", "精度", "准确率"],
    "参数": ["超参数", "权重", "系数", "变量"],
    "优化": ["改进", "提升", "增强", "调优"],
    "损失": ["误差", "代价", "目标函数", "Loss"],
    "监督": ["有监督", "有标签", "标注"],
    "交叉验证": ["K折验证", "交叉检验", "CV", "Cross Validation"],
    "向量": ["嵌入", "Embedding", "表征", "表示"],
    "检索": ["搜索", "查询", "召回", "匹配"],
    "生成": ["创建", "产生", "构建", "合成"],
    "Python": ["Python语言", "Python编程", "py"],
}

# English synonym dictionary
_EN_SYNONYMS = {
    "reset": ["initialize", "restore", "revert", "reinitialize"],
    "how": ["how to", "way to", "method for", "approach to"],
    "password": ["passphrase", "credential", "secret", "key"],
    "settings": ["configuration", "options", "preferences", "parameters"],
    "install": ["deploy", "setup", "configure", "set up"],
    "delete": ["remove", "clear", "uninstall", "purge"],
    "modify": ["change", "edit", "update", "alter"],
    "view": ["query", "display", "show", "browse"],
    "error": ["fault", "exception", "bug", "issue", "problem"],
    "connect": ["link", "attach", "join", "pair"],
    "start": ["launch", "begin", "initiate", "open"],
    "stop": ["halt", "terminate", "pause", "end", "quit"],
    "configure": ["setup", "arrange", "tune", "adjust"],
    "file": ["document", "data", "record", "archive"],
    "system": ["platform", "framework", "environment", "infrastructure"],
}


class QueryExpander:
    """Expand queries with synonyms and optional HyDE."""

    def __init__(
        self,
        synonym_enabled: bool = True,
        hyde_enabled: bool = False,
        hyde_model: str = "google-t5/t5-small",
        max_synonyms: int = 3,
    ):
        self.synonym_enabled = synonym_enabled
        self.hyde_enabled = hyde_enabled
        self.hyde_model = hyde_model
        self.max_synonyms = max_synonyms
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
        """
        result = {
            "original": query,
            "expanded": query,
            "hyde_text": None,
            "synonyms": [],
        }

        if self.synonym_enabled:
            synonyms = self._get_synonyms(query)
            result["synonyms"] = synonyms
            if synonyms:
                result["expanded"] = query + " " + " ".join(synonyms)

        if self.hyde_enabled:
            hyde_text = self._generate_hyde(query)
            if hyde_text:
                result["hyde_text"] = hyde_text

        return result

    def _get_synonyms(self, query: str) -> List[str]:
        """Get synonyms for words in the query (supports both Chinese and English)."""
        synonyms = []
        # Search Chinese synonyms
        for word, syns in _CN_SYNONYMS.items():
            if word in query:
                for s in syns[: self.max_synonyms]:
                    if s not in query:
                        synonyms.append(s)
        # Search English synonyms (case-insensitive)
        query_lower = query.lower()
        for word, syns in _EN_SYNONYMS.items():
            if word in query_lower:
                for s in syns[: self.max_synonyms]:
                    if s.lower() not in query_lower:
                        synonyms.append(s)
        return synonyms

    def _generate_hyde(self, query: str) -> Optional[str]:
        """Generate a hypothetical document using T5-small.

        This is disabled by default due to noise concerns.
        """
        if not self.hyde_enabled:
            return None

        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            if self._hyde_model is None:
                self._hyde_tokenizer = T5Tokenizer.from_pretrained(self.hyde_model)
                self._hyde_model = T5ForConditionalGeneration.from_pretrained(
                    self.hyde_model
                )

            input_text = f"Please write a passage to answer the question: {query}"
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
            hyde_text = self._hyde_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            return hyde_text
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"HyDE generation failed: {e}")
            return None