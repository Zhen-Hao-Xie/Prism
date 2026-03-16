"""
生成相关工具模块：包含文本生成时的辅助函数和类
"""

import torch
from transformers import StoppingCriteria
from typing import List, Union


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    关键词停止条件：当生成的文本中包含指定关键词时停止生成
    
    用法：
        stopping_criteria = StoppingCriteriaList([
            KeywordsStoppingCriteria(["</s>", "USER:"], tokenizer, input_ids)
        ])
        outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """
    
    def __init__(self, keywords: Union[str, List[str]], tokenizer, input_ids):
        """
        初始化关键词停止条件
        
        Args:
            keywords: 关键词或关键词列表
            tokenizer: 分词器
            input_ids: 输入的token IDs
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
        self.max_keyword_len = 0
        
        # 预处理关键词，转换为token IDs
        self.keyword_ids = []
        for keyword in keywords:
            keyword_ids = tokenizer(keyword).input_ids
            # 移除BOS token（如果存在）
            if len(keyword_ids) > 1 and keyword_ids[0] == tokenizer.bos_token_id:
                keyword_ids = keyword_ids[1:]
            self.max_keyword_len = max(self.max_keyword_len, len(keyword_ids))
            self.keyword_ids.append(torch.tensor(keyword_ids))

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        检查是否满足停止条件
        
        Returns:
            True表示应该停止生成，False表示继续
        """
        for i in range(output_ids.shape[0]):
            if self._check_single(output_ids[i].unsqueeze(0)):
                return True
        return False

    def _check_single(self, output_ids: torch.LongTensor) -> bool:
        """检查单个输出的停止条件"""
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        
        # 方法1：检查精确的token匹配
        for keyword_id in self.keyword_ids:
            keyword_id = keyword_id.to(output_ids.device)
            if keyword_id.shape[0] <= offset:
                if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                    return True
        
        # 方法2：解码后检查字符串包含
        if offset > 0:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, -offset:], 
                skip_special_tokens=True
            )[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        
        return False

