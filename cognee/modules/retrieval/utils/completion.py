from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.shared.logging_utils import get_logger  # Add this import

logger = get_logger()  # Initialize logger


async def generate_completion(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = render_prompt(user_prompt_path, args)
    system_prompt = read_query_prompt(system_prompt_path)

    # Log prompts
    logger.info(f"LLM system_prompt: {system_prompt}")
    logger.info(f"LLM user_prompt: {user_prompt}")

    llm_client = get_llm_client()
    try:
        result = await llm_client.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=str,
        )
        return result
    except Exception as e:
        logger.error(f"LLM acreate_structured_output error: {e}", exc_info=True)
        raise

async def generate_completion_with_function_calling(
    query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    tools: list = None,
) -> str:
    """Generates a completion using LLM with given context and prompts."""
    args = {"question": query, "context": context}
    user_prompt = render_prompt(user_prompt_path, args)
    system_prompt = read_query_prompt(system_prompt_path)

    # Log prompts
    logger.info(f"LLM system_prompt: {system_prompt}")
    logger.info(f"LLM user_prompt: {user_prompt}")

    llm_client = get_llm_client()
    try:
        result = await llm_client.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=str,
            tools=tools,
        )
        return result
    except Exception as e:
        logger.error(f"LLM acreate_structured_output error: {e}", exc_info=True)
        raise


async def summarize_text(
    text: str,
    prompt_path: str = "summarize_search_results.txt",
) -> str:
    """Summarizes text using LLM with the specified prompt."""
    system_prompt = read_query_prompt(prompt_path)
    llm_client = get_llm_client()

    return await llm_client.acreate_structured_output(
        text_input=text,
        system_prompt=system_prompt,
        response_model=str,
    )
