import os
from typing import Any, List, Optional

from cognee.shared.logging_utils import get_logger
from scrapegraph_py import Client

logger = get_logger("ScrapegraphTask")


async def scrape_urls(
    urls: List[str],
    user_prompt: str = "Extract the main content, title, and key information from this page",
    api_key: Optional[str] = None,
) -> List[dict]:
    """
    Scrape web content from a list of URLs using ScrapeGraphAI.

    Parameters
    ----------
    urls : List[str]
        List of URLs to scrape.
    user_prompt : str
        Natural language instruction describing what to extract from each page.
    api_key : Optional[str]
        ScrapeGraphAI API key. Falls back to the ``SGAI_API_KEY`` environment variable.

    Returns
    -------
    List[dict]
        List of dicts with ``url``, ``content``, and optionally ``error`` keys.
    """
    if api_key is None:
        api_key = os.getenv("SGAI_API_KEY")

    if not api_key:
        raise ValueError(
            "ScrapeGraphAI API key is required. Set the SGAI_API_KEY environment variable."
        )

    client = Client(api_key=api_key)
    results = []

    try:
        for url in urls:
            logger.info(f"Scraping URL: {url}")
            try:
                response = client.smartscraper(
                    website_url=url,
                    user_prompt=user_prompt,
                )
                results.append(
                    {
                        "url": url,
                        "content": response.get("result", ""),
                    }
                )
                logger.info(f"Successfully scraped: {url}")
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                results.append({"url": url, "content": "", "error": str(e)})
    finally:
        client.close()

    return results


async def scrape_and_add(
    urls: List[str],
    user_prompt: str = "Extract the main content, title, and key information from this page",
    api_key: Optional[str] = None,
    dataset_name: str = "scrapegraph",
) -> Any:
    """
    Scrape web content from a list of URLs and add it to cognee.

    Parameters
    ----------
    urls : List[str]
        List of URLs to scrape.
    user_prompt : str
        Natural language instruction describing what to extract from each page.
    api_key : Optional[str]
        ScrapeGraphAI API key. Falls back to the ``SGAI_API_KEY`` environment variable.
    dataset_name : str
        Name of the cognee dataset to add the scraped content to.

    Returns
    -------
    Any
        The cognee graph result after processing the scraped content.
    """
    import cognee

    scraped = await scrape_urls(urls=urls, user_prompt=user_prompt, api_key=api_key)

    successful = [item for item in scraped if not item.get("error")]
    if not successful:
        raise RuntimeError("No URLs were scraped successfully.")

    combined_text = "\n\n".join(
        f"Source: {item['url']}\n{item['content']}" for item in successful
    )

    await cognee.add(combined_text, dataset_name=dataset_name)
    result = await cognee.cognify()

    logger.info(f"Added {len(successful)} scraped pages to cognee dataset '{dataset_name}'")
    return result
