from typing import Any, Literal
from common.models.api_models import (
    GetActivationRequest,
    LayerOptimizerStateResponse,
    RunInfo,
    ActivationResponse,
    CompleteFileUploadResponse,
    FileUploadCompletionRequest,
    FileUploadRequest,
    LossReportRequest,
    MinerRegistrationResponse,
    OptimizerStateUpdate,
    RegisterMinerRequest,
    FileUploadResponse,
    SubmitActivationRequest,
    SubmittedWeightsAndOptimizerPresigned,
    SyncActivationAssignmentsRequest,
    WeightSubmitResponse,
    WeightUpdate,
    AttestationChallengeRequest,
    RequestAttestationChallengeResponse,
    SubmitMergedPartitionsRequest,
    MinerAttestationPayload,
)
from common.models.error_models import (
    LayerStateError,
    EntityNotRegisteredError,
    MinerFrozenError,
    MinerInitializingError,
    RunFullError,
    SpecVersionError,
)

from common.utils.exceptions import (
    APIException,
    LayerStateException,
    MinerFrozenException,
    MinerInitializingException,
    MinerNotRegisteredException,
    RunFullException,
    SpecVersionException,
)
from common.utils.partitions import MinerPartition
from common.utils.s3_utils import upload_parts
from common.utils.shared_states import LayerPhase
from loguru import logger
from subnet.common_api_client import CommonAPIClient
from substrateinterface.keypair import Keypair


class MinerAPIClient(CommonAPIClient):
    def __init__(self, hotkey: Keypair | None = None, is_mounted: bool = False, electron_version: str | None = None):
        self.hotkey = hotkey
        self.layer_state = LayerPhase.TRAINING
        self.is_mounted = is_mounted
        self.electron_version = electron_version

    async def fetch_run_info_request(self) -> list[RunInfo]:
        response = await CommonAPIClient.orchestrator_request(
            method="GET",
            path="/common/get_run_info",
            hotkey=self.hotkey,
            is_mounted=self.is_mounted,
            electron_version=self.electron_version,
        )
        parsed_response = self.parse_response(response)
        return [RunInfo.model_validate(run_info) for run_info in parsed_response]

    async def register_miner_request(
        self, register_miner_request: RegisterMinerRequest
    ) -> MinerRegistrationResponse | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/register",
                hotkey=self.hotkey,
                body=register_miner_request.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return MinerRegistrationResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error registering miner: {e}")
            raise

    async def change_payout_coldkey_request(self, payout_coldkey: str) -> dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/payout_coldkey",
                hotkey=self.hotkey,
                body={"payout_coldkey": payout_coldkey},
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return parsed_response
        except Exception as e:
            logger.error(f"Error changing payout coldkey: {e}")
            raise

    async def get_layer_state_request(self) -> LayerPhase | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/layer_state",
                hotkey=self.hotkey,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return LayerPhase(parsed_response)
        except Exception as e:
            logger.error(f"Error getting layer state: {e}")
            raise

    async def get_activations(self, get_activation_request: GetActivationRequest) -> list[ActivationResponse] | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_activations",
                hotkey=self.hotkey,
                body=get_activation_request.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return [ActivationResponse.model_validate(parsed_response) for parsed_response in parsed_response]
        except Exception as e:
            logger.error(f"Error getting activation: {e}")
            raise

    async def submit_weights(
        self,
        weight_update: WeightUpdate,
    ) -> WeightSubmitResponse:
        """Attempts to submit weights to the orchestrator.

        Returns:
            WeightSubmitResponse: Contains message and should_upload_optimizer_state flag.
                If should_upload_optimizer_state is True, the miner should upload their
                optimizer state and call submit_optimizer_state.
        """
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_weights",
                hotkey=self.hotkey,
                body=weight_update.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return WeightSubmitResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")
            raise

    async def submit_optimizer_state(
        self,
        optimizer_state_path: str,
    ) -> dict:
        """Submit optimizer state path to the orchestrator.

        This should only be called when submit_weights returns
        should_upload_optimizer_state=True.
        """
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_optimizer_state",
                hotkey=self.hotkey,
                body=OptimizerStateUpdate(optimizer_state_path=optimizer_state_path).model_dump(),
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error submitting optimizer state: {e}")
            raise

    async def get_layer_optimizer_state(self) -> LayerOptimizerStateResponse:
        """Get the layer optimizer state presigned URL for the miner's current layer.

        This is used by new miners joining a layer to download the optimizer state
        from one of the first miners who submitted weights.
        """
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_layer_optimizer_state",
                hotkey=self.hotkey,
            )
            parsed = self.parse_response(response)
            return LayerOptimizerStateResponse.model_validate(parsed)
        except Exception as e:
            logger.error(f"Error getting layer-based optimizer state: {e}")
            raise

    async def report_loss(self, loss_report: LossReportRequest) -> None:
        """Report loss to orchestrator"""
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/report_loss",
                hotkey=self.hotkey,
                body=loss_report.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            self.parse_response(response)
        except Exception as e:
            logger.error(f"Error reporting loss: {e}")
            raise e

    async def submit_activation_request(self, submit_activation_request: SubmitActivationRequest) -> None:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_activation",
                hotkey=self.hotkey,
                body=submit_activation_request.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            self.parse_response(response)
        except Exception as e:
            logger.error(f"Error submitting activation: {e}")
            raise

    async def sync_activation_assignments(self, activation_ids: list[str]) -> dict[str, bool]:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/sync_activation_assignments",
                hotkey=self.hotkey,
                body=SyncActivationAssignmentsRequest(activation_ids=activation_ids).model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error checking if activation is active: {e}")
            raise

    async def get_partitions(self) -> list[int] | dict:
        """Get the partition indices for a given hotkey."""
        try:
            response: list[MinerPartition] | dict = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_partitions",
                hotkey=self.hotkey,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            return self.parse_response(response)

        except Exception as e:
            logger.error(f"Error getting weight partition info: {e}")
            raise

    async def get_weight_path_per_layer(self) -> list[SubmittedWeightsAndOptimizerPresigned] | dict:
        """Get the weight path for a given layer."""
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_weight_path_per_layer",
                hotkey=self.hotkey,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            paths = [SubmittedWeightsAndOptimizerPresigned.model_validate(weight) for weight in parsed_response]
            return paths

        except Exception as e:
            logger.error(f"Error getting weight path per layer: {e}")
            raise

    async def notify_orchestrator_of_state_call(self) -> int | dict:
        """Notify the orchestrator of a state call."""
        try:
            response: int | dict = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/notify_orchestrator_of_state_call",
                hotkey=self.hotkey,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error notifying orchestrator of state call: {e}")
            raise

    async def get_learning_rate(self) -> float | dict:
        """Get the current learning rate."""
        try:
            response: float = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/learning_rate",
                hotkey=self.hotkey,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error getting learning rate: {e}")
            raise

    async def submit_merged_partitions(
        self,
        merged_partitions: list[MinerPartition],
        attestation: MinerAttestationPayload | None = None,
    ) -> dict:
        """Submit merged partitions to the orchestrator."""
        try:
            body = SubmitMergedPartitionsRequest(
                partitions=merged_partitions,
                attestation=attestation,
            ).model_dump()
            response: dict = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_merged_partitions",
                hotkey=self.hotkey,
                body=body,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error submitting merged partitions: {e}")
            raise

    async def request_attestation_challenge(
        self, action: Literal["weights", "merged_partitions"]
    ) -> RequestAttestationChallengeResponse | None:
        """Request a fresh attestation challenge for a specific action."""
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/request_attestation_challenge",
                hotkey=self.hotkey,
                body=AttestationChallengeRequest(action=action).model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed = self.parse_response(response)
            return RequestAttestationChallengeResponse.model_validate(parsed)
        except APIException as e:
            logger.warning(f"Attestation challenge unavailable for action {action}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error requesting attestation challenge for action {action}: {e}")
            raise

    async def initiate_file_upload_request(
        self,
        hotkey: Keypair,
        file_upload_request: FileUploadRequest,
    ) -> FileUploadResponse | FileUploadResponse | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/initiate_file_upload",
                hotkey=hotkey,
                body=file_upload_request.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return (
                FileUploadResponse.model_validate(parsed_response)
                if file_upload_request.multipart
                else FileUploadResponse.model_validate(parsed_response)
            )
        except Exception as e:
            logger.error(f"Error initiating file upload: {e}")
            raise

    @classmethod
    async def upload_to_s3(cls, urls: list[str], data: bytes, upload_id: str) -> list[dict] | None:
        assert len(urls) > 0, "No URLs provided"
        assert len(data) > 0, "No data provided"
        response = await upload_parts(urls=urls, data=data, upload_id=upload_id)
        return response

    async def complete_file_upload_request(
        self,
        hotkey: Keypair,
        file_upload_completion_request: FileUploadCompletionRequest,
    ) -> CompleteFileUploadResponse | dict:
        try:
            parts = file_upload_completion_request.parts or []
            logger.info(
                f"Sending complete_file_upload_request | "
                f"object_name={file_upload_completion_request.object_name} "
                f"upload_id={file_upload_completion_request.upload_id} "
                f"parts_count={len(parts)} "
                f"part_numbers={[p.get('PartNumber') for p in parts][:5]}"
            )
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/complete_multipart_upload",
                hotkey=hotkey,
                body=file_upload_completion_request.model_dump(),
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return CompleteFileUploadResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error completing file upload: {e}")
            raise

    async def get_merged_partitions(self, hotkey: Keypair) -> list[MinerPartition] | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/common/get_merged_partitions",
                hotkey=hotkey,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            parsed_response = self.parse_response(response)
            return [MinerPartition.model_validate(partition) for partition in parsed_response]
        except Exception as e:
            logger.error(f"Error getting merged partitions: {e}")
            raise

    def parse_response(self, response: Any) -> Any:
        """Parse the response from the orchestrator.
        If the response is a dictionary, check for error_name and handle the error accordingly.
        If the response is not a dictionary, return the response.
        """

        if not isinstance(response, dict):
            return response

        if (error_name := response.get("error_name")) is not None:
            if error_name == RunFullError.__name__:
                logger.error(f"Run is full: {response['error_dict']}")
                raise RunFullException(response["error_dict"]["message"])

            if error_name == LayerStateError.__name__:
                logger.warning(f"Layer state change: {response['error_dict']}")
                error_dict = LayerStateError(**response["error_dict"])
                self.layer_state = error_dict.actual_status
                raise LayerStateException(
                    f"Miner is moving state from {error_dict.expected_status} to {error_dict.actual_status}"
                )

            if error_name == EntityNotRegisteredError.__name__:
                logger.error(f"Miner not registered error: {response['error_dict']}")
                raise MinerNotRegisteredException("Miner not registered")

            if error_name == SpecVersionError.__name__:
                logger.error(f"Spec version mismatch: {response['error_dict']}")
                raise SpecVersionException(
                    expected_version=response["error_dict"]["expected_version"],
                    actual_version=response["error_dict"]["actual_version"],
                )

            if error_name == MinerFrozenError.__name__:
                logger.warning(f"Miner is frozen: {response['error_dict']}")
                raise MinerFrozenException(response["error_dict"]["message"])

            if error_name == MinerInitializingError.__name__:
                logger.warning(f"Miner is initializing: {response['error_dict']}")
                raise MinerInitializingException(response["error_dict"]["message"])

            else:
                raise Exception(f"Unexpected error from orchestrator. Response: {response}")

        else:
            return response

    async def get_previous_partitions(self, partition_indices: list[int]) -> list[MinerPartition] | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_previous_partitions",
                hotkey=self.hotkey,
                body=partition_indices,
                is_mounted=self.is_mounted,
                electron_version=self.electron_version,
            )
            response = self.parse_response(response)
            if response is None:
                logger.warning(f"No previous partitions found for partition indices: {partition_indices}")
            return [MinerPartition(**partition) for partition in response]

        except Exception as e:
            logger.error(f"Error getting previous partitions: {e}")
            raise
