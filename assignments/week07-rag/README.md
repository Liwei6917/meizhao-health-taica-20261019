# 每朝健康 · 第七週RAG作業

[開啟RAG02（主要作品）](https://colab.research.google.com/github/Liwei6917/meizhao-health-taica-20261019/blob/main/assignments/week07-rag/RAG02.ipynb) · [開啟RAG01（建立向量資料庫）](https://colab.research.google.com/github/Liwei6917/meizhao-health-taica-20261019/blob/main/assignments/week07-rag/RAG01.ipynb)

作者：李維｜411313779｜靜宜大學人工智慧學系3B

知識來源：本次自編《每朝健康：生活紀錄原型使用指南 v1.0》（AI協助）。17章，17個FAISS向量。不是醫療資料，也不是第五週網頁操作手冊。

## 開始使用
1. 開啟RAG02，使用CPU，選全部執行；首次安裝及下載公開模型需要數分鐘。
2. 第6節出現Gradio，預設使用免金鑰本機Qwen。問「加入生活小卡要做哪些步驟？」並核對右側來源。
3. 若選Groq，設定Colab密鑰 `Groq` 或 `GROQ_API_KEY`，並勾選資料傳送同意。金鑰不用交給助教。

RAG02會自動下載並驗證[固定版本rag-data.zip](https://raw.githubusercontent.com/Liwei6917/meizhao-health-taica-20261019/7b57d7a6281940da7fb30135a1895b7590d2301d/assignments/week07-rag/rag-data.zip)，不用先跑RAG01。RAG01可從嵌入的原始指南重建ZIP。

## 實際驗證
- 神經網路E5向量與FAISS建庫、重載成功，維度384。
- 8個問題的前三段命中章節：8/8。
- 本機Qwen真實選句JSON與對應原文已保存於筆記本，沒有以模擬回答冒充API。本機為模型選句、程式呈現原文；Groq模式才是自由解說。
- 程式驗證21項通過；本機測試不冒稱使用者Colab已執行。

小模型仍可能漏答，引用不是正確保證。原始輸出與限制保留供核對。

## 來源與協作
依老師公開[AI-Demo RAG01/RAG02](https://github.com/yenlung/AI-Demo)改作，使用E5、FAISS、LangChain、Transformers、AISuite與Gradio。影片連結是第六週角色與API課程；RAG以公開範例為依據。需求、命名與情境由使用者提出，指南、程式與測試由AI協助，供本人審閱。
