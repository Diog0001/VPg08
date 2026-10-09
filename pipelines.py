"""Три независимых пайплайна Haystack: индексация, поиск, суммаризация."""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from config import Settings
from prompts import ANSWER_TEMPLATE, SUMMARIZE_TEMPLATE

logger = logging.getLogger(__name__)


def chat_filter(chat_id: int | str) -> dict[str, Any]:
    return {
        "field": "meta.chat_id",
        "operator": "==",
        "value": str(chat_id),
    }


class PipelineManager:
    """Создаёт пайплайны один раз. Компоненты между пайплайнами не шарятся."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_key = Secret.from_token(settings.openai_api_key)
        self.document_store = PineconeDocumentStore(
            api_key=Secret.from_token(settings.pinecone_api_key),
            index=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec={
                "serverless": {
                    "region": settings.pinecone_region,
                    "cloud": settings.pinecone_cloud,
                }
            },
        )
        logger.info(
            "Pinecone: index=%s namespace=%s dimension=%s",
            settings.pinecone_index_name,
            settings.pinecone_namespace,
            settings.embedding_dimensions,
        )
        self.indexing_pipeline = self._create_indexing_pipeline()
        self.query_pipeline = self._create_query_pipeline()
        self.summarization_pipeline = self._create_summarization_pipeline()
        logger.info("Пайплайны Haystack собраны: indexing, query, summarization")

    def _openai_document_embedder(self) -> OpenAIDocumentEmbedder:
        return OpenAIDocumentEmbedder(
            api_key=self.api_key,
            api_base_url=self.settings.openai_base_url,
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
            meta_fields_to_embed=["username"],
        )

    def _openai_text_embedder(self) -> OpenAITextEmbedder:
        return OpenAITextEmbedder(
            api_key=self.api_key,
            api_base_url=self.settings.openai_base_url,
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )

    def _retriever(self) -> PineconeEmbeddingRetriever:
        return PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.settings.retrieval_top_k,
        )

    def _create_indexing_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("embedder", self._openai_document_embedder())
        pipeline.add_component(
            "writer",
            DocumentWriter(
                document_store=self.document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )
        pipeline.connect("embedder.documents", "writer.documents")
        return pipeline

    def _create_query_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self._openai_text_embedder())
        pipeline.add_component("retriever", self._retriever())
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        return pipeline

    def _create_summarization_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self._openai_text_embedder())
        pipeline.add_component("retriever", self._retriever())
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=SUMMARIZE_TEMPLATE,
                required_variables=["query", "documents", "dialogue"],
            ),
        )
        pipeline.add_component(
            "llm",
            OpenAIChatGenerator(
                api_key=self.api_key,
                api_base_url=self.settings.openai_base_url,
                model=self.settings.openai_model,
            ),
        )
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        return pipeline

    def index_message(
        self,
        *,
        chat_id: int,
        message_id: int,
        user_id: int,
        username: str,
        text: str,
        date_iso: str,
    ) -> None:
        content = (text or "").strip()
        if not content:
            return
        document = Document(
            id=f"tg-{chat_id}-{message_id}",
            content=content,
            meta={
                "user_id": str(user_id),
                "username": username,
                "chat_id": str(chat_id),
                "date": date_iso,
                "message_id": str(message_id),
            },
        )
        logger.info("Индексация message_id=%s chat_id=%s author=%s", message_id, chat_id, username)
        result = self.indexing_pipeline.run({"embedder": {"documents": [document]}})
        written = result.get("writer", {}).get("documents_written")
        logger.info("Индексация завершена, written=%s", written)

    def search(self, query: str, chat_id: int) -> list[Document]:
        logger.info("Поиск: chat_id=%s query=%r", chat_id, query[:120])
        result = self.query_pipeline.run(
            {
                "text_embedder": {"text": query},
                "retriever": {
                    "filters": chat_filter(chat_id),
                    "top_k": self.settings.retrieval_top_k,
                },
            }
        )
        documents = result.get("retriever", {}).get("documents", [])
        logger.info("Найдено документов: %s", len(documents))
        return documents

    def _generate(
        self,
        *,
        query: str,
        dialogue: str,
        chat_id: int,
        template: list,
    ) -> str:
        logger.info("LLM-пайплайн: chat_id=%s query=%r", chat_id, query[:120])
        result = self.summarization_pipeline.run(
            {
                "text_embedder": {"text": query},
                "retriever": {
                    "filters": chat_filter(chat_id),
                    "top_k": self.settings.retrieval_top_k,
                },
                "prompt_builder": {
                    "query": query,
                    "dialogue": dialogue or "(пусто)",
                    "template": template,
                },
            }
        )
        replies = result.get("llm", {}).get("replies", [])
        if not replies:
            logger.warning("LLM вернул пустой ответ: %s", result)
            return "Не удалось сформировать ответ. Попробуйте ещё раз."
        text = replies[0].text or ""
        logger.info("LLM ответ, символов: %s", len(text))
        return text.strip()

    def answer(self, query: str, dialogue: str, chat_id: int) -> str:
        return self._generate(
            query=query,
            dialogue=dialogue,
            chat_id=chat_id,
            template=ANSWER_TEMPLATE,
        )

    def summarize(self, dialogue: str, chat_id: int) -> str:
        return self._generate(
            query="Подведи итог этого обсуждения",
            dialogue=dialogue,
            chat_id=chat_id,
            template=SUMMARIZE_TEMPLATE,
        )
