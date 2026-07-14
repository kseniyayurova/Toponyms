from pathlib import Path
import gc
import json
import os
import re
import time

from typing import Any, Literal, TypedDict

import shutil
import subprocess

shutil.rmtree('iad2026-mapping-Russian-Songs', ignore_errors=True)

subprocess.run([
    "git", 
    "clone", 
    "https://github.com/kholodovTimur/iad2026-mapping-Russian-Songs.git"
], check=True)

PROJECT_ROOT = Path("/kaggle/working/iad2026-mapping-Russian-Songs")
PROMPTS_PATH = PROJECT_ROOT / "prompts.json"

with PROMPTS_PATH.open("r", encoding="utf-8") as f:
    jsonprompts = json.load(f)

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from pydantic import BaseModel, Field

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.location import Location

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from llama_cpp import Llama

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

from langgraph.graph import START, END, StateGraph

from nrclex import NRCLex

device = 0 if torch.cuda.is_available() else -1
if torch.cuda.is_available():
    torch.cuda.set_device(device)

from pydantic import PrivateAttr

class LlamaLLM(BaseChatModel):
    model_path: str

    _llm: Llama = PrivateAttr()

    def __init__(self, model_path: str, **kwargs: Any):
        llama_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"n_ctx", "n_batch", "n_threads", "verbose", "use_mmap"}
        }

        super().__init__(model_path=model_path)

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            use_mmap=False,
            **llama_kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def _convert_messages(self, messages: list) -> list[dict]:
        role_map = {
            "ai": "assistant",
            "human": "user",
            "system": "system",
        }

        return [
            {
                "role": role_map[message.type],
                "content": message.content,
            }
            for message in messages
        ]

    def _generate(self, messages: list, **kwargs: Any) -> ChatResult:
        response = self._llm.create_chat_completion(
            messages=self._convert_messages(messages),
            **kwargs,
        )

        content = response["choices"][0]["message"]["content"]

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=content)
                )
            ]
        )

    def with_structured_output(self, schema, include_raw: bool = False):
        def _generate_structured(messages: list, **kwargs: Any):
            try:
                response = self._llm.create_chat_completion(
                    messages=self._convert_messages(messages),
                    response_format={
                        "type": "json_object",
                        "schema": schema.model_json_schema(),
                    },
                    **kwargs,
                )
            except Exception as error:
                if include_raw:
                    return {
                        "raw": None,
                        "parsed": None,
                        "error": error,
                        "error_phase": "generation",
                    }
                raise

            content = response["choices"][0]["message"]["content"]

            try:
                parsed_response = schema.model_validate_json(content)
            except Exception as error:
                if include_raw:
                    return {
                        "raw": AIMessage(content=content),
                        "parsed": None,
                        "error": error,
                        "error_phase": "parsing",
                    }
                raise

            if include_raw:
                return {
                    "raw": AIMessage(content=content),
                    "parsed": parsed_response,
                    "error": None,
                    "error_phase": None,
                }

            return parsed_response

        return RunnableLambda(_generate_structured)

class Place(BaseModel):
    toponym: str = Field(
        description=jsonprompts["ToponymDescr"]
    )

    normal: str = Field(
        description=jsonprompts["NormalizDescr"]
    )

    type: Literal[
        "улица",
        "метро",
        "район",
        "город",
        "регион",
        "округ",
        "страна",
        "природа",
        "другое",
    ] = Field(
        description=jsonprompts["TypeTopDescr"]
    )


class SongInfo(BaseModel):
    places: list[Place] = Field(
        description=(
            "Все именованные географические объекты, найденные в тексте песни. "
            "Пустой список — если не найдено."
        )
    )


class AddressMatch(BaseModel):
    match: bool = Field(
        description="True, если адрес соответствует топониму в контексте песни."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность от 0.0 до 1.0.",
    )


class Song(TypedDict):
    song_text: str
    toponymns: SongInfo
    locations: list[Location]
    geo_objects: list[dict]
    tonality_dict: dict
    confident_topomymns: list[str]
    score: str
    recognitiontime: float
    geotime: float

class TonalityProceed:
    def __init__(
        self,
        chars_count: int = 1500,
        nrclex_emotions: list[str] | None = None,
    ):
        self.chars_count = chars_count

        self.emotions = nrclex_emotions or [
            "joy",
            "trust",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "anger",
            "anticipation",
        ]

        self.models = {
            "posneg_ru": pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                top_k=None,
                device=device,
            ),
            "five_emotion_ru": pipeline(
                "text-classification",
                model="cointegrated/rubert-tiny2-cedr-emotion-detection",
                top_k=None,
                device=device,
            ),
            "seven_emotion_ru": pipeline(
                "text-classification",
                model="Aniemore/rubert-tiny2-russian-emotion-detection",
                top_k=None,
                device=device,
            ),
        }

    def chunk_split(self, text: str, chars_count: int | None = None) -> list[str]:
        chars_count = chars_count or self.chars_count

        chunks = []
        start = 0

        while start < len(text):
            end = start + chars_count

            if end < len(text):
                space_index = text.rfind(" ", start, end)
                if space_index > start:
                    end = space_index

            chunks.append(text[start:end])
            start = end

        return chunks

    def text_split_proceed(
        self,
        model_pipeline,
        text: str,
        chars_count: int | None = None,
    ) -> dict:
        chars_count = chars_count or self.chars_count

        chunks = self.chunk_split(text, chars_count=chars_count)
        coeffs = [len(chunk) / len(text) for chunk in chunks]

        scores = {}

        for chunk_index, chunk in enumerate(chunks):
            result = model_pipeline(chunk)[0]

            for tonality in result:
                label = tonality["label"]
                score = tonality["score"]
                coeff = coeffs[chunk_index]

                scores.setdefault(label, []).append((score, coeff))

        return {
            label: sum(score * coeff for score, coeff in values)
            for label, values in scores.items()
        }

    def emotion_inspect(self, text_en: str) -> list[list[dict]]:
        lexicon = NRCLex()
        lexicon.load_raw_text(text_en)

        raw_scores = lexicon.raw_emotion_scores
        total_score = sum(raw_scores.get(emotion, 0) for emotion in self.emotions) or 1

        result = [
            {
                "label": emotion,
                "score": raw_scores.get(emotion, 0) / total_score,
            }
            for emotion in self.emotions
        ]

        return [result]

    def proceed_one_text(self, text_ru: str | None = None) -> dict:
        if not text_ru:
            return {"error": "Текст пуст"}

        clean_text = re.sub(r"\s+", " ", text_ru).strip()

        state = {
            "models_time": {},
            "models_result": {},
            "error": None,
        }

        for model_name, model_pipeline in self.models.items():
            start_time = time.time()

            try:
                state["models_result"][model_name] = self.text_split_proceed(
                    model_pipeline,
                    clean_text,
                )
            except Exception as error:
                state["error"] = f"{error}. Model with error: {model_name}"
                return state

            state["models_time"][model_name] = time.time() - start_time

        torch.cuda.empty_cache()
        gc.collect()

        return state

class SongToponymGeoRecognition:
    def __init__(
        self,
        local = True,
        recognition_model_repo_id: str = "unsloth/Qwen3.6-35B-A3B-GGUF",
        recognition_model_name: str = "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
        geo_model_repo_id: str = "unsloth/Qwen3.6-35B-A3B-GGUF",
        geo_model_name: str = "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
        recognition_mt: int = 512,
        recognition_t: float = 0.7,
        recog_ctx: int = 4096,
        geo_mt: int = 512,
        geo_t: float = 0.1,
        geo_ctx: int = 4096,
        **kwargs,
    ):
        self.local = local
        self.recognition_model_repo_id = recognition_model_repo_id
        self.recognition_model_name = recognition_model_name
        self.geo_model_repo_id = geo_model_repo_id
        self.geo_model_name = geo_model_name

        self.recognition_mt = recognition_mt
        self.recognition_t = recognition_t
        self.recog_ctx = recog_ctx

        self.geo_mt = geo_mt
        self.geo_t = geo_t
        self.geo_ctx = geo_ctx

        self.tonality_finder = TonalityProceed()

        self.geolocator = Nominatim(user_agent="iad_project")
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=1,
        )

        self.model = self.graph_instance(**kwargs)

    def get_model(
        self,
        model_repo_id: str,
        model_name: str,
        mt: int,
        t: float,
        ctx: int,
        **kwargs,
    ):
        if self.local == False:
            return init_chat_model(model=model_name, temperature=t)
        else:
            model_path = hf_hub_download(
                repo_id=model_repo_id,
                filename=model_name,
            )
    
            return LlamaLLM(
                model_path=model_path,
                n_ctx=ctx,
                **kwargs,
            )

    def get_toponyms(self, model, state: Song) -> Song:
        start_time = time.time()

        coder = model.with_structured_output(SongInfo)

        state["toponymns"] = coder.invoke(
            [
                SystemMessage(content="Отвечай без размышлений. /no_think"),
                HumanMessage(
                    content=(
                        f"{jsonprompts['MainToponymPrompt']}\n"
                        f"Вот текст песни: {state['song_text']}"
                    )
                ),
            ],
            max_tokens=self.recognition_mt,
            temperature=self.recognition_t,
        )

        state["recognitiontime"] = time.time() - start_time

        return state

    def get_context(self, song_text: str, toponym: str) -> str:
        toponym_index = song_text.find(toponym)

        if toponym_index == -1:
            return song_text

        lines = song_text.split("\n")

        if len(lines) > 10:
            char_count = 0
            target_line = 0

            for line_index, line in enumerate(lines):
                if char_count + len(line) >= toponym_index:
                    target_line = line_index
                    break

                char_count += len(line) + 1

            start = max(0, target_line - 5)
            end = min(len(lines), target_line + 6)

            return "\n".join(lines[start:end])

        start = max(0, toponym_index - 150)
        end = min(len(song_text), toponym_index + len(toponym) + 150)

        return song_text[start:end]

    def get_geo(self, model, state: Song) -> Song:
        start_time = time.time()

        locations = []
        confident_topomymns = []
        geo_objects = []

        coder = model.with_structured_output(AddressMatch)
        prompt = jsonprompts["MainGeoPrompt"]

        for toponym in state["toponymns"].model_dump()["places"]:
            toponym_location = self.geocode(
                toponym["normal"],
                geometry="geojson",
                exactly_one=True,
            )

            if toponym_location is None:
                continue

            context = self.get_context(
                song_text=state["song_text"],
                toponym=toponym["toponym"],
            )

            answer = coder.invoke(
                [
                    SystemMessage(content="Отвечай без размышлений. /no_think"),
                    HumanMessage(
                        content=prompt.format(
                            context=context,
                            toponym=toponym["normal"],
                            address=toponym_location.address,
                        )
                    ),
                ],
                max_tokens=self.geo_mt,
                temperature=self.geo_t,
            )

            if answer.match:
                locations.append(toponym_location)
                confident_topomymns.append(toponym["toponym"])
                geo_objects.append(
                    {
                        "toponym": toponym["toponym"],
                        "normal": toponym["normal"],
                        "type": toponym["type"],
                        "address": toponym_location.address,
                        "latitude": float(toponym_location.latitude),
                        "longitude": float(toponym_location.longitude),
                        "geometry": toponym_location.raw.get("geojson"),
                    }
                )

        state["locations"] = locations
        state["geo_objects"] = geo_objects
        state["confident_topomymns"] = confident_topomymns
        state["geotime"] = time.time() - start_time

        return state

    def get_tonality(self, state: Song) -> Song:
        state["tonality_dict"] = self.tonality_finder.proceed_one_text(
            state["song_text"]
        )

        return state

    def should_continue(self, state: Song):
        if len(state["toponymns"].places) == 0:
            state["confident_topomymns"] = []
            state["geo_objects"] = []
            return END

        return "get_geo"

    def graph_instance(self, **kwargs):
        same_model = (
            self.recognition_model_repo_id == self.geo_model_repo_id
            and self.recognition_model_name == self.geo_model_name
        )

        if same_model:
            recognition_model = self.get_model(
                model_repo_id=self.recognition_model_repo_id,
                model_name=self.recognition_model_name,
                mt=self.recognition_mt,
                t=self.recognition_t,
                ctx=self.recog_ctx,
                **kwargs,
            )
            geo_model = recognition_model
        else:
            recognition_model = self.get_model(
                model_repo_id=self.recognition_model_repo_id,
                model_name=self.recognition_model_name,
                mt=self.recognition_mt,
                t=self.recognition_t,
                ctx=self.recog_ctx,
                **kwargs,
            )
            geo_model = self.get_model(
                model_repo_id=self.geo_model_repo_id,
                model_name=self.geo_model_name,
                mt=self.geo_mt,
                t=self.geo_t,
                ctx=self.geo_ctx,
                **kwargs,
            )

        graph_builder = StateGraph(Song)

        graph_builder.add_node(
            "get_toponyms",
            lambda state: self.get_toponyms(recognition_model, state),
        )
        graph_builder.add_node(
            "get_geo",
            lambda state: self.get_geo(geo_model, state),
        )
        graph_builder.add_node(
            "get_tonality",
            lambda state: self.get_tonality(state),
        )

        graph_builder.add_edge(START, "get_toponyms")

        graph_builder.add_conditional_edges(
            "get_toponyms",
            self.should_continue,
            ["get_geo", END],
        )

        graph_builder.add_edge("get_geo", "get_tonality")
        graph_builder.add_edge("get_tonality", END)

        return graph_builder.compile()

    def proceed_one_track(self, song: str):
        callback = UsageMetadataCallbackHandler()
        return self.model.invoke(
            {
                "song_text": song,
            }, config={"callbacks": [callback]}
        ), callback


