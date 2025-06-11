# Summary : 
# The code outlines a comprehensive framework for integrating Azure's SearchClient and OpenAI's AsyncOpenAI in a chat application. 
# It features classes and methods for handling user queries, generating search queries, retrieving documents, and constructing responses 
# based on documents fetched from Azure's search services and OpenAI's conversational model.

import os
import re
import json
import logging
from dataclasses import dataclass
from utils import list_abbreviations
from typing import Any, Optional, List, Dict, Union, cast, Coroutine, AsyncGenerator

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    VectorQuery,
    RawVectorQuery,
    CaptionResult,
    QueryType,
)

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)

from integration_azure.messagebuilder import MessageBuilder
from integration_azure.authentication import AuthenticationHelper
from integration_azure.modelhelper import get_token_limit

@dataclass
class Document:
    id: Optional[str]
    content: Optional[str]
    embedding: Optional[List[float]]
    image_embedding: Optional[List[float]]
    category: Optional[str]
    sourcepage: Optional[str]
    sourcefile: Optional[str]
    oids: Optional[List[str]]
    groups: Optional[List[str]]
    captions: List[CaptionResult]

    def serialize_for_results(self) -> dict[str, Any]:

        return {
            "id": self.id,
            "content": self.content,
            "embedding": Document.trim_embedding(self.embedding),
            "imageEmbedding": Document.trim_embedding(self.image_embedding),
            "category": self.category,
            "sourcepage": self.sourcepage,
            "sourcefile": self.sourcefile,
            "oids": self.oids,
            "groups": self.groups,
            "captions": [
                {
                    "additional_properties": caption.additional_properties,
                    "text": caption.text,
                    "highlights": caption.highlights,
                }
                for caption in self.captions
            ]
            if self.captions
            else [],
        }

    @classmethod
    def trim_embedding(cls, embedding: Optional[List[float]]) -> Optional[str]:
        # print("ENTERED: Approach::trim_embedding()...")

        """Returns a trimmed list of floats from the vector embedding."""
        if embedding:
            if len(embedding) > 2:
                # Format the embedding list to show the first 2 items followed by the count of the remaining items."""
                return f"[{embedding[0]}, {embedding[1]} ...+{len(embedding) - 2} more]"
            else:
                return str(embedding)

        return None

@dataclass
class ThoughtStep:
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None


class RagResponder:
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    query_prompt_few_shots = [
        {"role": USER, "content": "How to request for SAP system access?"},
        {"role": ASSISTANT, "content": "Raise SSR following clause 4.2 of ITP-CORP-001 Access Control Policy"},
        {"role": USER, "content": "Can I use Facebook?"},
        {"role": ASSISTANT, "content": "Social media sites / applications are prohibited as per clause 5.5.4 of ITP-CORP-004 IT Resources and Acceptable Use Policy"},
        {"role": USER, "content": "How to request for an application system change?"},
        {"role": ASSISTANT, "content": "Discuss the nature of the change request before raising an SSR following the ITP-CORP-007 Corporate System Service Request Policy"},
        {"role": ASSISTANT, "content": "The health screening benefits for a (M4 and above) include comprehensive health evaluations, personalized health plans, and priority access to health services, as detailed on page 113 of Employee Handbook."},
        {"role": USER, "content": "What are the health screening benefits for a General Counsel/General Manager/Vice President/Senior Vice President/Chief Executive Officer?"}, 
        {"role": ASSISTANT, "content": "The health screening benefits for a (G1 and above) include comprehensive health evaluations, personalized health plans, and priority access to health services, as detailed on page 113 of Employee Handbook."},
        {"role": USER, "content": "What are the health screening benefits for a Section Manager I/Staff Engineer I/Staff Executive I/Staff System Analyst I/Executive Assistant II/Section Manager II/Staff Engineer II/Staff Planner II/Manager/Principal Engineer/Deputy Director/Senior Manager/Senior Principal Engineer?"}, 
        {"role": ASSISTANT, "content": "The health screening benefits for a (M1-M3) only include comprehensive health evaluations, personalized health plans, and priority access to health services, as detailed on page 113 of Employee Handbook."}
    ]
    NO_RESPONSE = "0"

    follow_up_questions_prompt_content = """Generate 3 very brief follow-up questions that the user would likely ask next while following the below rules.
    MANDATORILY only genearate questions such that it can be answered using the provided `Sources` in the current conversation. DONT generate questions that cannot be answered from the `Sources`.
    Enclose the follow-up questions in double angle brackets. Example:
    <<What should I do if I need urgent privileged access?>>
    <<What is a System Service Request (SSR) and how do I submit one?>>
    <<How do I contact the IT Security Team?>>
    Do not repeat questions that have already been asked.
    Make sure the last question ends with ">>".
    """

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge.
    You have access to Azure AI Search index with 100's of documents.
    Generate a search query based on the conversation and the new question.
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Do not include any special characters like '+'.
    If the question is not in English, translate the question to English before generating the search query.
    If you cannot generate a search query, return just the number 0.
    """

    abbreviations = '\n'.join(list_abbreviations())

    # confidence_prompt = """
    # You retrieved this context: {context}. The question is: {question}.
    # Before even answering the question, consider whether you have sufficient information in the context to answer the question fully.
    # Your output should JUST be the boolean true or false, of if you have sufficient information in the article to answer the question.
    # Respond with just one word, the boolean true or false. You must output the word 'True', or the word 'False', nothing else.
    # """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    @property
    def system_message_chat_conversation(self):
            return """You are an assistant who helps the UTAC company employees with their IT/HR Policy questions. You do not answer any other off-topic questions.
            When a user types common greetings such as "hello", "hi", "good morning", "okay", "sure", "thanks", etc., respond with small talk and ask how you can assist them and be professional and courteous. Ignore the listed sources for these instances and do not reply with an "Insufficient information" response. 
            For all other inquiries, be complete in your answers. DON'T provide misleading answers that are open to interpretation or assumptions. Answer only the latest question.
            Answer ONLY with the facts listed in the list of sources below and provide all necessary steps required to answer. If you are ABSOLUTELY SURE that there isn't enough information below, then only say:

            "I'm sorry, I don't have enough information to resolve that right now.
            Please raise a ticket here: [click here](http://sg7srk2papp03.sg.utacgroup.com:10779/)
            Is there anything else I can assist you with?"

            After every answer, append:
            "Is there anything else I can help with? Would you like to raise a ticket? [click here](http://sg7srk2papp03.sg.utacgroup.com:10779/)"
            except if you are responding with the insufficient information message above, in which case do not add this footer.

            For tabular information, convert it into a hierarchical/flat structure using unordered bullet points depending on the table, ensuring that ALL data is included in the response representing all rows and columns with breakup and row-span and col-span if available. Do not return tables in the response. Do not return markdown format. 
            If the question is not in English, identify the language and MANDATORILY answer in the language used in the question. Only give answer not the language in response.
            Each source has a name followed by a colon and the actual information. Use bold or heading style for section headers, then bullet points for steps. Do not add a number then a linebreak for section titles.
            MANDATORILY Use square brackets to reference the source, for example [1. ITP-CORP-001 Access Control Policy.docx_Protected.pdf]. Only reference the same source once, Don't combine sources, list each source separately, for example [1. ITP-CORP-001 Access Control Policy.docx_Protected.pdf] [2. ITP-CORP-010 Data Management Policy.docx_Protected.pdf]
        <Abbreviations>
        {abbreviations}
        </Abbreviations>
        {follow_up_questions_prompt}
        {injected_prompt}
        """

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        # print("ENTERED: Approach::build_filter()...")

        exclude_category = overrides.get("exclude_category") or None
        country = overrides.get("country") or None
        site = overrides.get("site") or None
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
        if security_filter:
            filters.append(security_filter)
        if country:
            filters.append(f"(country/any(c: c eq '{country}' or c eq '{country.lower()}'))")
        if site:
            filters.append(f"(site/any(s: s eq '{site}' or s eq '{site.lower()}'))")
        return None if len(filters) == 0 else " and ".join(filters)

    def get_messages_from_history(self, system_prompt: str, model_id: str, history: List[Dict[str, str]], user_content: Union[str, List[ChatCompletionContentPartParam]], max_tokens: int, few_shots=[]) -> List[ChatCompletionMessageParam]:
        """
        Builds a message list from conversation history for generating a query or response.
        """
        message_builder = MessageBuilder(system_prompt, model_id)

        # Reverse few_shots to mimic conversation order
        for shot in reversed(few_shots):
            message_builder.insert_message(shot.get("role"), shot.get("content"))

        append_index = len(few_shots) + 1

        message_builder.insert_message("user", user_content, index=append_index)
        total_token_count = message_builder.count_tokens_for_message(dict(message_builder.messages[-1]))

        # Process the rest of the conversation history
        for message in reversed(history[:-1]):
            potential_message_count = message_builder.count_tokens_for_message(message)
            if (total_token_count + potential_message_count) > max_tokens:
                logging.debug("Reached max tokens of %d, history will be truncated", max_tokens)
                break
            message_builder.insert_message(message["role"], message["content"], index=append_index)
            total_token_count += potential_message_count

        return message_builder.messages

    def get_search_query(self, chat_completion: ChatCompletion, user_query: str):
        """
        Extracts the search query from the chat completion response or uses the original user query if no optimized query is found.
        """
        response_message = chat_completion.choices[0].message
        if function_call := response_message.function_call:
            if function_call.name == "search_sources":
                arg = json.loads(function_call.arguments)
                search_query = arg.get("search_query", self.NO_RESPONSE)
                if search_query != self.NO_RESPONSE:
                    return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    async def search(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
    ) -> List[Document]:
        # print("ENTERED: Approach::search()...")

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if use_semantic_ranker and query_text:
            results = await self.search_client.search(
                search_text=query_text,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=vectors,
            )
        else:
            results = await self.search_client.search(
                search_text=query_text or "", filter=filter, top=top, vector_queries=vectors
            )

        documents = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        embedding=document.get("embedding"),
                        image_embedding=document.get("imageEmbedding"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(List[CaptionResult], document.get("@search.captions")),
                    )
                )
        print("num documents", len(documents))
        return documents

    def get_sources_content(self, results: List[Document], use_semantic_captions: bool, use_image_citation: bool) -> List[str]:
        """
        Generates content strings with citations for each document in the search results.
        """
        if use_semantic_captions:
            return [
                f"{self.get_citation(doc.sourcepage, use_image_citation)}: {doc.content}"
                for doc in results
            ]
        else:
            return [
                f"{self.get_citation(doc.sourcepage, use_image_citation)}: {doc.content}"
                for doc in results
            ]

    def get_citation(self, sourcepage: str, use_image_citation: bool) -> str:
        """
        Generates a citation string based on the source page.
        """
        if use_image_citation:
            return sourcepage
        else:
            path, ext = os.path.splitext(sourcepage)
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                page_number = int(path[page_idx + 1:])
                return f"{path[:page_idx]}.pdf#page={page_number}"
            return sourcepage

    async def compute_text_embedding(self, q: str):
        embedding = await self.openai_client.embeddings.create(
            # Azure Open AI takes the deployment name as the model name
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=q,
        )
        query_vector = embedding.data[0].embedding
        return RawVectorQuery(vector=query_vector, k=50, fields="embedding")

    def get_system_prompt(self, override_prompt: Optional[str], follow_up_questions_prompt: str) -> str:
        # print("ENTERED: ChatApproach::get_system_prompt()...")

        if override_prompt is None:
            return self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt, abbreviations=self.abbreviations
            )
        elif override_prompt.startswith(">>>"):
            return self.system_message_chat_conversation.format(
                injected_prompt=override_prompt[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt, abbreviations=self.abbreviations
            )
        else:
            return override_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt)

    def extract_followup_questions(self, content: str):
        # print("ENTERED: ChatApproach::extract_followup_questions()...")
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    def _build_citation_map(self, docs: List[Document]) -> Dict[int, Dict[str, str]]:
        """
        Build a mapping of citation numbers to source details for footer citations.
        """
        citation_map = {}
        for idx, doc in enumerate(docs, start=1):
            if doc.content and doc.sourcepage:
                citation_map[idx] = {
                    'sourcepage': doc.sourcepage,
                    'content_preview': doc.content.strip()[:100] + "..." if len(doc.content.strip()) > 100 else doc.content.strip()
                }
        return citation_map

    def _generate_footer_citations(self, response_text: str, citation_map: Dict[int, Dict[str, str]]) -> str:
        """
        Generate footer citations based on citations found in the response.
        """
        # Find all citation numbers in the response (e.g., [1. filename], [2. filename], etc.)
        citation_pattern = r'\[(\d+)\.\s*([^\]]+)\]'
        found_citations = re.findall(citation_pattern, response_text)
        
        if not found_citations:
            return ""
        
        # Extract unique citation numbers
        citation_numbers = set()
        for citation_num, _ in found_citations:
            citation_numbers.add(int(citation_num))
        
        if not citation_numbers:
            return ""
        
        # Build footer citations
        footer_lines = ["\n\n---", "**Sources:**"]
        
        for citation_num in sorted(citation_numbers):
            if citation_num in citation_map:
                source_info = citation_map[citation_num]
                footer_lines.append(f"[{citation_num}] {source_info['sourcepage']}")
        
        return "\n".join(footer_lines)

    def _create_citation_chunk(self, footer_text: str) -> Dict[str, Any]:
        """
        Create a mock chunk object for footer citations that matches the streaming format.
        """
        return {
            "choices": [
                {
                    "delta": {"content": footer_text},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "object": "chat.completion.chunk",
        }

    async def _rag_stream(self, docs, user_query, stream):
        """
        Streams a response from the OpenAI Chat API using both system prompt and sources,
        including citations and follow-up/ticket offers as per UTAC policy.
        """
        # Build the sources context with citation info
        sources_content = []
        for idx, doc in enumerate(docs, start=1):
            if doc.content and doc.sourcepage:
                # Compose source string (reference index and source page)
                sources_content.append(
                    f"[{idx}. {doc.sourcepage}] {doc.content.strip()}"
                )
        sources_text = "\n".join(sources_content)

        # Compose the user message including sources as context
        user_content = (
            f"{user_query}\n\nSources:\n{sources_text}"
            if sources_text else user_query
        )

        # Use your system message
        system_prompt = self.system_message_chat_conversation.format(
            abbreviations=self.abbreviations,
            follow_up_questions_prompt=self.follow_up_questions_prompt_content,
            injected_prompt=""
        )

        # Compose messages for chat completion
        messages = [
            {"role": self.SYSTEM, "content": system_prompt},
            {"role": self.USER, "content": user_content}
        ]

        # Call the OpenAI API with streaming enabled
        stream_response = await self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment or self.chatgpt_model,
            messages=messages,
            stream=True,
            max_tokens=1024,         # You can adjust if needed
            temperature=0.0,
        )

        # Stream the chunks as they come in
        async for chunk in stream_response:
            yield chunk


    async def run_until_final_call(
            self,
            messages,
            overrides: dict[str, Any],
            auth_claims: dict[str, Any],
            should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:

        # print("ENTERED: RagResponder::run_until_final_call()...")

        logging.debug("Running the RagResponder with messages.")

        # has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_text = True
        # has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        has_vector = True

        # use_semantic_captions = True if overrides.get("semantic_captions") and has_text else False
        use_semantic_captions = True

        top = overrides.get("top", 3)
        filter = self.build_filter(overrides, auth_claims)
        print("filter is", filter)
        # use_semantic_ranker = True if overrides.get("semantic_ranker") and has_text else False
        use_semantic_ranker = True

        original_user_query = messages[-1]["content"]
        print("original_user_query = ",original_user_query)

        user_query_request = "Generate search query for: " + original_user_query

        functions = [
            {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            }
        ]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        messages = self.get_messages_from_history(
            system_prompt=self.query_prompt_template,
            model_id=self.chatgpt_model,
            history=messages,
            user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - len(user_query_request),
            few_shots=self.query_prompt_few_shots,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=messages,  # type: ignore
            # Azure Open AI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,
            max_tokens=100,  # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            functions=functions,
            function_call="auto",
            seed=123
        )

        query_text = self.get_search_query(chat_completion, original_user_query)
        print("Optimized: query_text for top-K chunk search = ", query_text)

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        print("has_vector = ", has_vector)
        print("has_text = ", has_text)
        print("use_semantic_ranker = ", use_semantic_ranker)

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(query_text))

        # Only keep the text query if the retrieval mode uses text, otherwise drop it
        if not has_text:
            query_text = None

        results = await self.search(
            top, query_text, filter, vectors, use_semantic_ranker, use_semantic_captions
        )

        logging.debug(f"RAG_DEBUG: Retrieved {len(results)} documents")

        # Track if we retrieved any documents for later decision making
        results_found = bool(results)
        if not results_found:
            logging.warning("RAG_DEBUG: 0 docs — falling back to plain OpenAI chat")

        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)
        #print(f"content which come from reterival: ",content)
        # # STEP 2.5: Check confidence in case of insufficient information

        # messages_confidence = [
        #     {
        #         "role": "user",
        #         "content": self.confidence_prompt.format(
        #             context=content, question=original_user_query
        #         ),
        #     }
        # ]

        # chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
        #     messages=messages_confidence,  # type: ignore
        #     # Azure Open AI takes the deployment name as the model name
        #     model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
        #     temperature=0.0,
        #     max_tokens=10,  # Setting too low risks malformed JSON, setting too high may affect performance
        #     n=1,
        #     seed=123,
        #     logprobs=True,
        #     top_logprobs=2
        # )

        # print("confidence")
        # print(chat_completion.json())

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 1024
        messages_token_limit = self.chatgpt_token_limit - response_token_limit
        messages.pop(0)
        messages_new = self.get_messages_from_history(
            system_prompt=system_message,
            model_id=self.chatgpt_model,
            history=messages,
            # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve follow up questions prompt.
            user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=messages_token_limit,
        )


        print("messages_new: ",messages_new)
        data_points = {"text": sources_content}

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Original user query",
                    original_user_query,
                ),
                ThoughtStep(
                    "Generated search query",
                    query_text,
                    {"use_semantic_captions": use_semantic_captions, "has_vector": has_vector},
                ),
                ThoughtStep("Results", [result.serialize_for_results() for result in results]),
                ThoughtStep("Prompt", [str(message) for message in messages_new]),
            ],
            # Add citation map to extra_info for use in streaming
            "citation_map": self._build_citation_map(results) if results else {},
        }

        if should_stream and results_found:
            logging.info("RAG_DEBUG: Returning RAG stream coroutine")

            async def _wrapper():
                return self._rag_stream(
                    docs=results,
                    user_query=original_user_query,
                    stream=True,
                )

            return extra_info, _wrapper()

        logging.info(
            f"RAG_DEBUG: Preparing to call OpenAI completions.create. stream={should_stream}"
        )
        logging.info(f"RAG_DEBUG: OpenAI client type: {type(self.openai_client)}")

        chat_coroutine = self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment or self.chatgpt_model,
            messages=messages_new,
            temperature=overrides.get("temperature", 0.0),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
            seed=123,
        )

        logging.debug(f"RAG_DEBUG: chat_coroutine assigned: {repr(chat_coroutine)}")

        return extra_info, chat_coroutine
    
    async def run_without_streaming(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            history, overrides, auth_claims, should_stream=False
        )
        # print("ENTERED: ChatApproach::run_without_streaming()...")

        chat_completion_response: ChatCompletion = await chat_coroutine
        chat_resp = chat_completion_response.model_dump()  # Convert to dict to make it JSON serializable
        chat_resp["choices"][0]["context"] = extra_info
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(chat_resp["choices"][0]["message"]["content"])
            chat_resp["choices"][0]["message"]["content"] = content
            chat_resp["choices"][0]["context"]["followup_questions"] = followup_questions
        chat_resp["choices"][0]["session_state"] = session_state
        return chat_resp

    async def run_with_streaming(
        self,
        history: list[dict[str, str]],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:

        logging.info("RAG_DEBUG: Entered run_with_streaming.")

        extra_info, chat_coroutine = await self.run_until_final_call(
            history, overrides, auth_claims, should_stream=True
        )

        # 1) Yield the initial assistant role + context chunk
        yield {
            "choices": [
                {
                    "delta": {"role": self.ASSISTANT},
                    "context": extra_info,
                    "session_state": session_state,
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "object": "chat.completion.chunk",
        }
        logging.info("RAG_DEBUG: Awaiting chat_coroutine to get async stream")
        stream_iterator = await chat_coroutine

        logging.info(f"RAG_DEBUG: Streaming from {type(stream_iterator)}")

        # 4) Stream every chunk from that iterator
        followup_questions_started = False
        followup_content = ""

        async for event_chunk in stream_iterator:
            event = event_chunk.model_dump()  # pydantic -> dict
            if event["choices"]:
                content = event["choices"][0]["delta"].get("content") or ""
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        event["choices"][0]["delta"]["content"] = earlier_content
                        yield event
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield event

        # 5) After the loop, emit any aggregated follow-ups
        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {
                "choices": [
                    {
                        "delta": {"role": self.ASSISTANT},
                        "context": {"followup_questions": followup_questions},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "object": "chat.completion.chunk",
            }

    async def run(
        self, messages: list[dict], stream: bool = False, session_state: Any = None, context: dict[str, Any] = {}
    ) -> Union[dict[str, Any], AsyncGenerator[dict[str, Any], None]]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})

        # print("ENTERED: RagResponder::RUN()...")

        # print("RagResponder: streaming is True.")
        if stream:
            # print("ChatApproach: streaming is False.")
            return self.run_with_streaming(messages, overrides, auth_claims, session_state)
        else:
            # print("ChatApproach: streaming is True.")
            return await self.run_without_streaming(messages, overrides, auth_claims, session_state)
