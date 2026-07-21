import asyncio
from aiohttp import ClientSession, ClientTimeout
from common import settings as common_settings
from common.models.api_models import ValidationTaskResponse, ValidatorRegistrationResponse, SubnetScores, ValidatorTask
from common.settings import ORCHESTRATOR_HOST, ORCHESTRATOR_PORT, ORCHESTRATOR_SCHEMA
from common.utils.epistula import create_message_body, generate_header
from common.utils.exceptions import APIException, RateLimitException
from loguru import logger
from bittensor_wallet import Keypair

from subnet.common_api_client import CommonAPIClient

HEADER_REQUEST_ID = "X-Request-Id"


class ValidatorAPIClient(CommonAPIClient):
    @classmethod
    async def register_validator_request(cls, hotkey: Keypair) -> ValidatorRegistrationResponse | dict:
        try:
            response = await cls.orchestrator_request(method="POST", path="/validator/register", hotkey=hotkey)
            if hasattr(response, "error_name"):
                return response

            return ValidatorRegistrationResponse(**response)
        except Exception as e:
            logger.error(f"Error registering validator: {e}")
            raise e

    @classmethod
    async def get_global_miner_scores(cls, hotkey: Keypair) -> SubnetScores | None:
        """Get the global scores for all miners from the orchestrator."""
        try:
            response: SubnetScores = await cls.orchestrator_request(
                method="GET", path="/validator/global_miner_scores", hotkey=hotkey
            )
            if hasattr(response, "error_name"):
                logger.error(f"Error getting global miner scores: {response}")
                return
            return response

        except Exception as e:
            logger.error(f"Error getting global miner scores: {e}")
            raise e

    @classmethod
    async def submit_miner_scores(cls, hotkey: Keypair, miner_scores: dict[str, dict[str, float | str]]) -> dict:
        """Submit the miner scores to the orchestrator.

        Args:
            hotkey (Keypair): The hotkey of the validator.
            miner_scores (dict[str, dict[str, float | str]]): The miner scores to submit. dict[str(uid) : dict{task_type : score, hotkey : str}]
        """
        try:
            response: dict = await cls.orchestrator_request(
                method="POST", path="/validator/submit_miner_scores", hotkey=hotkey, body=miner_scores
            )
            if hasattr(response, "error_name"):
                return response
            return response
        except Exception as e:
            logger.error(f"Error submitting miner scores: {e}")
            raise e

    @classmethod
    async def fetch_task(cls, hotkey: Keypair) -> ValidatorTask | dict:
        try:
            response: ValidatorTask = await cls.orchestrator_request(
                method="GET", path="/validator/fetch_task", hotkey=hotkey
            )
            logger.info(f"Fetched task: {response}")
            # if hasattr(response, "error_name"):
            #     logger.error(f"Error fetching task: {response}")
            #     return None

            return ValidatorTask(**response) if response else None
        except Exception as e:
            logger.error(f"Error fetching task: {e}")
            raise e

    @classmethod
    async def submit_task_result(cls, hotkey: Keypair, task_result: ValidationTaskResponse) -> ValidationTaskResponse:
        try:
            response = await cls.orchestrator_request(
                method="POST", path="/validator/submit_task_result", hotkey=hotkey, body=task_result.model_dump()
            )
            if hasattr(response, "error_name"):
                return response
            return response
        except Exception as e:
            logger.error(f"Error submitting task result: {e}")
            raise e

    @classmethod
    async def get_validator_code(cls, hotkey: Keypair) -> bytes:
        """Get validator code as a ZIP file (binary data)."""
        path = "/validator/get_validator_code"
        logger.opt(colors=True).debug(f"\n<magenta>Making orchestrator request | method: GET | path: {path}</magenta>")

        headers = None
        response = None
        request_id = None
        body_bytes = create_message_body(data={})
        response_text = None

        if hotkey:
            headers = generate_header(hotkey, body_bytes)

        for i in range(common_settings.REQUEST_RETRY_COUNT):
            try:
                if i:
                    logger.warning(f"Retrying request to endpoint {path} (attempt {i + 1})")

                timeout = ClientTimeout(total=common_settings.CLIENT_REQUEST_TIMEOUT)
                async with ClientSession(timeout=timeout) as session:
                    async with session.request(
                        "GET",
                        f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}{path}",
                        headers=headers,
                    ) as response:
                        # Extract request ID from response headers
                        request_id = response.headers.get(HEADER_REQUEST_ID, "unknown")

                        # Add request ID to logger context for all subsequent logs
                        with logger.contextualize(request_id=request_id):
                            if response.status == 429:
                                logger.warning(f"Rate limited on request to endpoint {path}")
                                await asyncio.sleep(2**i)
                                continue

                            if response.status != 200:
                                # Handle non-JSON error responses first
                                response_text = await response.text()
                                msg = f"{response.status} - {response_text}"
                                raise APIException(f"Error making orchestrator request to endpoint {path}: {msg}")

                            # Success - read binary data instead of JSON
                            logger.info(f"Validator code requested {hotkey.ss58_address[:8]}")
                            response_bytes = await response.read()
                            logger.info(f"Validator code received, size: {len(response_bytes)} bytes")
                            logger.debug(
                                f"Successfully completed request to {path}; response size: {len(response_bytes)} bytes"
                            )
                            return response_bytes
            except RateLimitException:
                logger.error("Rate limit exception, applying exponential backoff")
                await asyncio.sleep(2**i)
            except Exception as e:
                logger.error(f"Error getting validator code: {e}")
                await asyncio.sleep(1)

        # The only time you get here is because you've exhausted all retries.
        error_msg = f"Failed request after {common_settings.REQUEST_RETRY_COUNT} attempts: {response.status if response else 'No response'}, {response_text if response_text else 'No response text'}"
        if request_id:
            with logger.contextualize(request_id=request_id):
                logger.error(error_msg)
        raise RateLimitException(error_msg)
