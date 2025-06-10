import dataclasses
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import AsyncGenerator, cast

from azure.core.exceptions import ResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.keyvault.secrets.aio import SecretClient
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.storage.blob.aio import BlobServiceClient
from datetime import datetime
from openai import APIError, AsyncAzureOpenAI, AsyncOpenAI
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from quart import (
    Blueprint,
    Quart,
    abort,
    current_app,
    jsonify,
    make_response,
    request,
    send_file,
    send_from_directory,
)
from quart_cors import cors

from app.backend.approaches.approach import Approach
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.chatreadretrievereadvision import ChatReadRetrieveReadVisionApproach
from approaches.retrievethenread import RetrieveThenReadApproach
from approaches.retrievethenreadvision import RetrieveThenReadVisionApproach
from core.authentication import AuthenticationHelper
from rag_response_generator.azure_rag_responder import RagResponder
import uuid
from azure.identity.aio import ClientSecretCredential
import httpx


CONFIG_OPENAI_TOKEN = "openai_token"
CONFIG_CREDENTIAL = "azure_credential"
CONFIG_ASK_APPROACH = "ask_approach"
CONFIG_ASK_VISION_APPROACH = "ask_vision_approach"
CONFIG_CHAT_VISION_APPROACH = "chat_vision_approach"
CONFIG_CHAT_APPROACH = "chat_approach"
CONFIG_RAGRESPONSEGEN_APPROACH = "azure_rag_response_gen_approach"
CONFIG_BLOB_CONTAINER_CLIENT = "blob_container_client"
CONFIG_AUTH_CLIENT = "auth_client"
CONFIG_GPT4V_DEPLOYED = "gpt4v_deployed"
CONFIG_SEARCH_CLIENT = "search_client"
CONFIG_OPENAI_CLIENT = "openai_client"
CONFIG_AZURE_COSMOSDB_CLIENT = "cosmos_client"
ERROR_MESSAGE = """The app encountered an error processing your request.
If you are an administrator of the app, view the full error in the logs. See aka.ms/appservice-logs for more information.
Error type: {error_type}
"""
ERROR_MESSAGE_FILTER = """Your message contains content that was flagged by the OpenAI content filter."""

bp = Blueprint("routes", __name__, static_folder="static")
# Fix Windows registry issue with mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


@bp.route("/")
async def index():
    return await bp.send_static_file("index.html")


# Empty page is recommended for login redirect to work.
# See https://github.com/AzureAD/microsoft-authentication-library-for-js/blob/dev/lib/msal-browser/docs/initialization.md#redirecturi-considerations for more information
@bp.route("/redirect")
async def redirect():
    return ""


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory(Path(__file__).resolve().parent / "static" / "assets", path)


# Serve content files from blob storage from within the app to keep the example self-contained.
# *** NOTE *** this assumes that the content files are public, or at least that all users of the app
# can access all the files. This is also slow and memory hungry.
@bp.route("/content/<path>")
async def content_file(path: str):
    # Remove page number from path, filename-1.txt -> filename.txt
    if path.find("#page=") > 0:
        path_parts = path.rsplit("#page=", 1)
        path = path_parts[0]
    logging.info("Opening file %s at page %s", path)
    blob_container_client = current_app.config[CONFIG_BLOB_CONTAINER_CLIENT]
    try:
        blob = await blob_container_client.get_blob_client(path).download_blob()
    except ResourceNotFoundError:
        logging.exception("Path not found: %s", path)
        abort(404)
    if not blob.properties or not blob.properties.has_key("content_settings"):
        abort(404)
    mime_type = blob.properties["content_settings"]["content_type"]
    if mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    blob_file = io.BytesIO()
    await blob.readinto(blob_file)
    blob_file.seek(0)
    return await send_file(blob_file, mimetype=mime_type, as_attachment=False, attachment_filename=path)


def error_dict(error: Exception) -> dict:
    if isinstance(error, APIError) and error.code == "content_filter":
        return {"error": ERROR_MESSAGE_FILTER}
    return {"error": ERROR_MESSAGE.format(error_type=type(error))}


def error_response(error: Exception, route: str, status_code: int = 500):
    logging.exception("Exception in %s: %s", route, error)
    if isinstance(error, APIError) and error.code == "content_filter":
        status_code = 400
    return jsonify(error_dict(error)), status_code


@bp.route("/ask", methods=["POST"])
async def ask():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    try:
        context["auth_claims"] = await auth_helper.get_auth_claims_if_enabled(request.headers)
        use_gpt4v = context.get("overrides", {}).get("use_gpt4v", False)
        approach: Approach
        if use_gpt4v and CONFIG_ASK_VISION_APPROACH in current_app.config:
            approach = cast(Approach, current_app.config[CONFIG_ASK_VISION_APPROACH])
        else:
            approach = cast(Approach, current_app.config[CONFIG_ASK_APPROACH])
        r = await approach.run(
            request_json["messages"], context=context, session_state=request_json.get("session_state")
        )

        prompt_text = request_json["messages"][-1]["content"]
        ai_reply = r["choices"][0]["message"]["content"]
        session_id = request_json.get("session_state")
        category = await categorize_question(prompt_text)
        await log_to_sharepoint(prompt_text, ai_reply, session_id, category)

        return jsonify(r)
    except Exception as error:
        return error_response(error, "/ask")

async def categorize_question(question):
    openai_client = current_app.config[CONFIG_OPENAI_CLIENT]
    chat_completion = await openai_client.chat.completions.create(
        model= os.environ["AZURE_OPENAI_CHATGPT_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": "You are an AI assistant that categorizes questions into predefined categories."},
            {"role": "user", "content": f"Categorize this question into one of the following categories: Vacation Related, Travel, IT Policy, HR, Other. Question: {question}"}
        ],
        temperature=0.3,
        max_tokens=100
    )
    return chat_completion.choices[0].message.content.strip()


async def log_to_sharepoint(question: str, response: str, session_id: str | None, category: str):
    """Persist a Q&A record to the configured SharePoint list."""
    timestamp = datetime.utcnow().isoformat()

    tenant_id = os.getenv("SHAREPOINT_TENANT_ID")
    client_id = os.getenv("SHAREPOINT_CLIENT_ID")
    client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET")
    if not (tenant_id and client_id and client_secret):
        raise RuntimeError("Azure AD credentials not set in env variables")

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    scope = os.getenv("SHAREPOINT_SCOPE", "https://graph.microsoft.com/.default")
    token_response = await credential.get_token(scope)
    access_token = token_response.token

    sp_graph_url = os.getenv("SHAREPOINT_GRAPH_URL")
    sp_library_name = os.getenv("SHAREPOINT_LIBRARY_NAME")
    if not (sp_graph_url and sp_library_name):
        raise RuntimeError("SharePoint Graph URL or Library name not set")

    payload = {
        "fields": {
            "Title": timestamp,
            "Question": question,
            "Response": response,
            "Category": category,
            "Timestamp": timestamp,
            "SessionID": session_id or "",
        }
    }

    url = f"{sp_graph_url}/lists/{sp_library_name}/items"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()


@bp.route("/save", methods=["POST"])
async def save():
    # 1) Generate ID + Timestamp
    new_id = str(uuid.uuid4())
    try:
        if not request.is_json:
            return jsonify({"error": "request must be json"}), 415

        new_item = await request.get_json()
        new_item["id"] = new_id
        new_item["timestamp"] = datetime.utcnow().isoformat()

        # 2) Categorize the question
        question = new_item.get("question", "")
        category = await categorize_question(question)
        new_item["category"] = category

        # 3) Acquire a Graph API access token via client‐credentials
        tenant_id = os.getenv("SHAREPOINT_TENANT_ID")
        client_id = os.getenv("SHAREPOINT_CLIENT_ID")
        client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET")
        if not (tenant_id and client_id and client_secret):
            raise RuntimeError("Azure AD credentials not set in env variables")

        # Use the async ClientSecretCredential to fetch a token for Microsoft Graph
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        # This scope is required for Graph service calls
        scope = os.getenv("SHAREPOINT_SCOPE", "https://graph.microsoft.com/.default")
        token_response = await credential.get_token(scope)
        access_token = token_response.token

        # 4) Build the JSON payload to create a new List item
        #    (Make sure the field-names match your SP List's internal field names exactly.)
        sp_graph_url = os.getenv("SHAREPOINT_GRAPH_URL")
        sp_library_name = os.getenv("SHAREPOINT_LIBRARY_NAME")
        if not (sp_graph_url and sp_library_name):
            raise RuntimeError("SharePoint Graph URL or Library name not set")

        fields_payload = {
            "fields": {
                # Title is required for any SharePoint List item
                "Title": new_item["timestamp"],

                # Map your columns exactly as defined in your List
                "Question": question,
                "Category": category,
                "Timestamp": new_item["timestamp"],
                "Response": new_item.get("response_text", ""),
                "SessionID": new_item.get("session_id", ""),


                # If you have other columns (e.g. UserID, Response), add them here:
                # "UserID": new_item.get("user_id", ""),
                # "Response": new_item.get("response_text", ""),
                # etc.
            }
        }

        graph_url = f"{sp_graph_url}/lists/{sp_library_name}/items"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            create_resp = await client.post(graph_url, json=fields_payload, headers=headers)
            create_resp.raise_for_status()
            created_item = create_resp.json()

        # 5) Return Graph’s response JSON to the caller
        return jsonify(created_item), 201

    except Exception as e:
        logging.exception("Exception while saving to SharePoint: %s", e)
        return jsonify({"error": str(e)}), 500

@bp.route("/reindex", methods=["POST"])
async def reindex():
    try:
        if not request.is_json:
            return jsonify({"error": "request must be json"}), 415
        payload = await request.get_json()
        return jsonify(payload)
    except Exception as error:
        logging.exception("Exception while saving: %s", error)
        return error_response(error, "/save")

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


async def format_as_ndjson(r: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    try:
        async for event in r:
            yield json.dumps(event, ensure_ascii=False, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("Exception while generating response stream: %s", error)
        yield json.dumps(error_dict(error))


@bp.route("/chat", methods=["POST"])
async def chat():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    try:
        context["auth_claims"] = await auth_helper.get_auth_claims_if_enabled(request.headers)
        use_gpt4v = context.get("overrides", {}).get("use_gpt4v", False)
        # approach: Approach
        # if use_gpt4v and CONFIG_CHAT_VISION_APPROACH in current_app.config:
        #     approach = cast(Approach, current_app.config[CONFIG_CHAT_VISION_APPROACH])
        # else:
        #     approach = cast(Approach, current_app.config[CONFIG_CHAT_APPROACH])

        approach = current_app.config[CONFIG_RAGRESPONSEGEN_APPROACH]

        print("request_json = ",request_json)

        result = await approach.run(
            request_json["messages"],
            stream=request_json.get("stream", False),
            context=context,
            session_state=request_json.get("session_state"),
        )

        print("result = ",result)

        prompt_text = request_json["messages"][-1]["content"]
        session_id = request_json.get("session_state")
        category = await categorize_question(prompt_text)

        if isinstance(result, dict):
            ai_reply = result["choices"][0]["message"]["content"]
            await log_to_sharepoint(prompt_text, ai_reply, session_id, category)
            return jsonify(result)
        else:
            async def log_wrapper(gen):
                full_reply = ""
                async for event in gen:
                    if event.get("choices"):
                        choice = event["choices"][0]
                        delta = choice.get("delta", {})
                        if "content" in delta and delta["content"]:
                            full_reply += delta["content"]
                        message = choice.get("message", {})
                        if "content" in message and message["content"]:
                            full_reply += message["content"]
                    yield event
                await log_to_sharepoint(prompt_text, full_reply, session_id, category)

            response = await make_response(format_as_ndjson(log_wrapper(result)))
            response.timeout = None  # type: ignore
            response.mimetype = "application/json-lines"
            return response
    except Exception as error:
        return error_response(error, "/chat")


# Send MSAL.js settings to the client UI
@bp.route("/auth_setup", methods=["GET"])
def auth_setup():
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    return jsonify(auth_helper.get_auth_setup_for_client())


@bp.route("/config", methods=["GET"])
async def config():
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    print("here", request.headers)
    ad_info = await auth_helper.get_country_and_site(request.headers)
    return jsonify({
        "showGPT4VOptions": current_app.config[CONFIG_GPT4V_DEPLOYED],
        "ad_info": ad_info
        })


@bp.before_app_serving
async def setup_clients():
    # Replace these with your own values, either in environment variables or directly here
    AZURE_STORAGE_ACCOUNT = os.environ["AZURE_STORAGE_ACCOUNT"]
    AZURE_STORAGE_CONTAINER = os.environ["AZURE_STORAGE_CONTAINER"]
    AZURE_SEARCH_SERVICE = os.environ["AZURE_SEARCH_SERVICE"]
    AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
    VISION_SECRET_NAME = os.getenv("VISION_SECRET_NAME")
    AZURE_KEY_VAULT_NAME = os.getenv("AZURE_KEY_VAULT_NAME")
    # Shared by all OpenAI deployments
    OPENAI_HOST = os.getenv("OPENAI_HOST", "azure")
    OPENAI_CHATGPT_MODEL = os.environ["AZURE_OPENAI_CHATGPT_MODEL"]
    OPENAI_EMB_MODEL = os.getenv("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002")
    # Used with Azure OpenAI deployments
    AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
    AZURE_OPENAI_GPT4V_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT4V_DEPLOYMENT")
    AZURE_OPENAI_GPT4V_MODEL = os.environ.get("AZURE_OPENAI_GPT4V_MODEL")
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT") if OPENAI_HOST == "azure" else None
    AZURE_OPENAI_EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT") if OPENAI_HOST == "azure" else None
    AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
    # Used only with non-Azure OpenAI deployments
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")

    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    AZURE_USE_AUTHENTICATION = os.getenv("AZURE_USE_AUTHENTICATION", "").lower() == "true"
    AZURE_ENFORCE_ACCESS_CONTROL = os.getenv("AZURE_ENFORCE_ACCESS_CONTROL", "").lower() == "true"
    AZURE_SERVER_APP_ID = os.getenv("AZURE_SERVER_APP_ID")
    AZURE_SERVER_APP_SECRET = os.getenv("AZURE_SERVER_APP_SECRET")
    AZURE_CLIENT_APP_ID = os.getenv("AZURE_CLIENT_APP_ID")
    AZURE_AUTH_TENANT_ID = os.getenv("AZURE_AUTH_TENANT_ID", AZURE_TENANT_ID)

    KB_FIELDS_CONTENT = os.getenv("KB_FIELDS_CONTENT", "content")
    KB_FIELDS_SOURCEPAGE = os.getenv("KB_FIELDS_SOURCEPAGE", "sourcepage")

    AZURE_SEARCH_QUERY_LANGUAGE = os.getenv("AZURE_SEARCH_QUERY_LANGUAGE", "en-us")
    AZURE_SEARCH_QUERY_SPELLER = os.getenv("AZURE_SEARCH_QUERY_SPELLER", "lexicon")

    AZURE_COSMOSDB_ACCOUNT_NAME = os.getenv("AZURE_COSMOSDB_ACCOUNT_NAME")

    USE_GPT4V = os.getenv("USE_GPT4V", "").lower() == "true"

    # Use the current user identity to authenticate with Azure OpenAI, AI Search and Blob Storage (no secrets needed,
    # just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
    # keys for each service
    # If you encounter a blocking error during a DefaultAzureCredential resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
    azure_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    # Set up clients for AI Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        credential=azure_credential,
    )
    search_index_client = SearchIndexClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        credential=azure_credential,
    )
    blob_client = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net", credential=azure_credential
    )
    blob_container_client = blob_client.get_container_client(AZURE_STORAGE_CONTAINER)

    # Set up authentication helper
    auth_helper = AuthenticationHelper(
        search_index=(await search_index_client.get_index(AZURE_SEARCH_INDEX)) if AZURE_USE_AUTHENTICATION else None,
        use_authentication=AZURE_USE_AUTHENTICATION,
        server_app_id=AZURE_SERVER_APP_ID,
        server_app_secret=AZURE_SERVER_APP_SECRET,
        client_app_id=AZURE_CLIENT_APP_ID,
        tenant_id=AZURE_AUTH_TENANT_ID,
        require_access_control=AZURE_ENFORCE_ACCESS_CONTROL,
    )



    vision_key = None
    if VISION_SECRET_NAME and AZURE_KEY_VAULT_NAME:  # Cognitive vision keys are stored in keyvault
        key_vault_client = SecretClient(
            vault_url=f"https://{AZURE_KEY_VAULT_NAME}.vault.azure.net", credential=azure_credential
        )
        vision_secret = await key_vault_client.get_secret(VISION_SECRET_NAME)
        vision_key = vision_secret.value
        await key_vault_client.close()

    # Used by the OpenAI SDK
    openai_client: AsyncOpenAI

    if OPENAI_HOST == "azure":
        token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
        # Store on app.config for later use inside requests
        openai_client = AsyncAzureOpenAI(
            api_version="2023-07-01-preview",
            azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
            azure_ad_token_provider=token_provider,
        )
    else:
        openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_ORGANIZATION,
        )

    current_app.config[CONFIG_OPENAI_CLIENT] = openai_client
    current_app.config[CONFIG_SEARCH_CLIENT] = search_client
    current_app.config[CONFIG_BLOB_CONTAINER_CLIENT] = blob_container_client
    current_app.config[CONFIG_AUTH_CLIENT] = auth_helper

    current_app.config[CONFIG_GPT4V_DEPLOYED] = bool(USE_GPT4V)

    # Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
    # or some derivative, here we include several for exploration purposes
    current_app.config[CONFIG_ASK_APPROACH] = RetrieveThenReadApproach(
        search_client=search_client,
        openai_client=openai_client,
        auth_helper=auth_helper,
        chatgpt_model=OPENAI_CHATGPT_MODEL,
        chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        embedding_model=OPENAI_EMB_MODEL,
        embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
        sourcepage_field=KB_FIELDS_SOURCEPAGE,
        content_field=KB_FIELDS_CONTENT,
        query_language=AZURE_SEARCH_QUERY_LANGUAGE,
        query_speller=AZURE_SEARCH_QUERY_SPELLER,
    )

    # current_app.config[CONFIG_AZURE_COSMOSDB_CLIENT] = cosmos_client

    if USE_GPT4V:
        if vision_key is None:
            raise ValueError("Vision key must be set (in Key Vault) to use the vision approach.")

        current_app.config[CONFIG_ASK_VISION_APPROACH] = RetrieveThenReadVisionApproach(
            search_client=search_client,
            openai_client=openai_client,
            blob_container_client=blob_container_client,
            auth_helper=auth_helper,
            vision_endpoint=AZURE_VISION_ENDPOINT,
            vision_key=vision_key,
            gpt4v_deployment=AZURE_OPENAI_GPT4V_DEPLOYMENT,
            gpt4v_model=AZURE_OPENAI_GPT4V_MODEL,
            embedding_model=OPENAI_EMB_MODEL,
            embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
            sourcepage_field=KB_FIELDS_SOURCEPAGE,
            content_field=KB_FIELDS_CONTENT,
            query_language=AZURE_SEARCH_QUERY_LANGUAGE,
            query_speller=AZURE_SEARCH_QUERY_SPELLER,
        )

        current_app.config[CONFIG_CHAT_VISION_APPROACH] = ChatReadRetrieveReadVisionApproach(
            search_client=search_client,
            openai_client=openai_client,
            blob_container_client=blob_container_client,
            auth_helper=auth_helper,
            vision_endpoint=AZURE_VISION_ENDPOINT,
            vision_key=vision_key,
            gpt4v_deployment=AZURE_OPENAI_GPT4V_DEPLOYMENT,
            gpt4v_model=AZURE_OPENAI_GPT4V_MODEL,
            embedding_model=OPENAI_EMB_MODEL,
            embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
            sourcepage_field=KB_FIELDS_SOURCEPAGE,
            content_field=KB_FIELDS_CONTENT,
            query_language=AZURE_SEARCH_QUERY_LANGUAGE,
            query_speller=AZURE_SEARCH_QUERY_SPELLER,
        )

    current_app.config[CONFIG_CHAT_APPROACH] = ChatReadRetrieveReadApproach(
        search_client=search_client,
        openai_client=openai_client,
        auth_helper=auth_helper,
        chatgpt_model=OPENAI_CHATGPT_MODEL,
        chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        embedding_model=OPENAI_EMB_MODEL,
        embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
        sourcepage_field=KB_FIELDS_SOURCEPAGE,
        content_field=KB_FIELDS_CONTENT,
        query_language=AZURE_SEARCH_QUERY_LANGUAGE,
        query_speller=AZURE_SEARCH_QUERY_SPELLER
    )

    current_app.config[CONFIG_RAGRESPONSEGEN_APPROACH] = RagResponder(
        search_client=search_client,
        openai_client=openai_client,
        auth_helper=auth_helper,
        chatgpt_model=OPENAI_CHATGPT_MODEL,
        chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        embedding_model=OPENAI_EMB_MODEL,
        embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
        sourcepage_field=KB_FIELDS_SOURCEPAGE,
        content_field=KB_FIELDS_CONTENT,
        query_language=AZURE_SEARCH_QUERY_LANGUAGE,
        query_speller=AZURE_SEARCH_QUERY_SPELLER,
    )

@bp.after_app_serving
async def close_clients():
    await current_app.config[CONFIG_SEARCH_CLIENT].close()
    await current_app.config[CONFIG_BLOB_CONTAINER_CLIENT].close()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)

    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        configure_azure_monitor()
        # This tracks HTTP requests made by aiohttp:
        AioHttpClientInstrumentor().instrument()
        # This tracks HTTP requests made by httpx/openai:
        HTTPXClientInstrumentor().instrument()
        # This middleware tracks app route requests:
        app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)  # type: ignore[method-assign]

    # Level should be one of https://docs.python.org/3/library/logging.html#logging-levels
    default_level = "INFO"  # In development, log more verbosely
    if os.getenv("WEBSITE_HOSTNAME"):  # In production, don't log as heavily
        default_level = "WARNING"
    logging.basicConfig(level=os.getenv("APP_LOG_LEVEL", default_level))

    if allowed_origin := os.getenv("ALLOWED_ORIGIN"):
        app.logger.info("CORS enabled for %s", allowed_origin)
        cors(app, allow_origin=allowed_origin, allow_methods=["GET", "POST"])
    return app
