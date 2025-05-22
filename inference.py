import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    pipeline
)
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from peft import PeftModel


class Bot:
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    EMBEDDING_MODEL = "cointegrated/LaBSE-en-ru"

    def __init__(
        self,
        model_path: str,
        docs_path: str,
        index_path: str,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.docs_path = docs_path
        self.index_path = index_path
        self.device = device

        # 1) Строим/загружаем retrieval-индекс
        if not os.path.isdir(self.index_path) or not os.listdir(self.index_path):
            self._build_retrieval_index()
        self._load_retriever()

        # 2) Загружаем модель, токенизатор и создаём цепочку QA
        self._load_model_and_chain()

    def _build_retrieval_index(self):
        loader = DirectoryLoader(self.docs_path, glob="**/*")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        store = FAISS.from_documents(texts, embeddings)
        os.makedirs(self.index_path, exist_ok=True)
        store.save_local(self.index_path)

    def _load_retriever(self):
        embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        store = FAISS.load_local(
            self.index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = store.as_retriever(search_kwargs={"k": 3})

    def _load_model_and_chain(self):
        # 2.1) Базовая модель + LORA
        base = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            trust_remote_code=True,
            device_map=self.device
        )
        peft = PeftModel.from_pretrained(
            base,
            self.model_path,
            trust_remote_code=True,
            device_map=self.device
        )
        model = peft.merge_and_unload()

        # 2.2) Токенизатор и пайплайн
        tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=0 if self.device != "cpu" else -1,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            return_full_text=False,
            # clean_up_tokenization_spaces=True
        )
        llm = HuggingFacePipeline(pipeline=gen_pipe)

        # 2.3) Формирование prompt-а
        def format_prompt(context, question):
            messages = [
                {"role": "system", "content":
                 "Ты — полезный AI-ассистент. Отвечай на вопросы, используя предоставленный контекст. "
                 "Если информации нет в контексте, скажи об этом и предложи свой вариант ответа."},
                {"role": "user", "content":
                 f"Контекст:\n{context}\n\nВопрос: {question}"}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=format_prompt("{context}", "{question}")
        )

        # 2.4) Сборка диалоговой цепочки с памятью
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False
        )

    def run(self, question: str) -> str:
        """Единственный метод, вызываемый для получения ответа."""
        result = self.qa_chain({"question": question})
        return result["answer"]


if __name__ == "__main__":
    bot = Bot(
        model_path="./sft_checkpoints_ru_lora/",
        docs_path="./docs/",
        index_path="./vector_index_ru/",
        device="cpu"
    )

    questions = [
        "Какое расстояние от Земли до Луны?",
        "Как подключиться к API нашего продукта?",
        "Какие ограничения по длине записи в системе?",
    ]
    for q in questions:
        print(f"\nВопрос: {q}")
        print("Ответ:", bot.run(q))

    while True:
        q = input("Вопрос: ")
        print("Ответ:", bot.run(q))
