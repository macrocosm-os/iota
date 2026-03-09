"""Tests for P2P timing detail integration in the stats tracker."""

from common.iroh.timings import P2POperationTimings
from miner.utils.stats import ActivationTiming, ActivationTimingStage, P2PTimingDetail, StatsTracker


class TestP2PTimingDetail:
    def test_from_operation_timings(self):
        op = P2POperationTimings(
            connection_duration=0.05,
            stream_open_duration=0.01,
            send_duration=0.1,
            receive_duration=0.3,
            total_start=1000.0,
            total_end=1000.46,
            total_duration=0.46,
            attempt_count=2,
            retry_count=1,
            total_backoff_time=0.25,
            errors=["RuntimeError: flaky (attempt 1)"],
            bytes_sent=128,
            bytes_received=8192,
        )
        detail = P2PTimingDetail.from_operation_timings(op)

        assert detail.connection_duration == 0.05
        assert detail.stream_open_duration == 0.01
        assert detail.send_duration == 0.1
        assert detail.receive_duration == 0.3
        assert detail.total_start == 1000.0
        assert detail.total_end == 1000.46
        assert detail.total_duration == 0.46
        assert detail.attempt_count == 2
        assert detail.retry_count == 1
        assert detail.total_backoff_time == 0.25
        assert detail.errors == ["RuntimeError: flaky (attempt 1)"]
        assert detail.bytes_sent == 128
        assert detail.bytes_received == 8192

    def test_from_empty_operation_timings(self):
        detail = P2PTimingDetail.from_operation_timings(P2POperationTimings())
        assert detail.connection_duration is None
        assert detail.attempt_count == 0
        assert detail.errors == []

    def test_errors_list_independence(self):
        """Ensure from_operation_timings copies the errors list."""
        op = P2POperationTimings(errors=["e1"])
        detail = P2PTimingDetail.from_operation_timings(op)
        detail.errors.append("e2")
        assert len(op.errors) == 1

    def test_serialises_to_dict(self):
        detail = P2PTimingDetail(
            connection_duration=0.05,
            send_duration=0.1,
            attempt_count=1,
        )
        d = detail.model_dump()
        assert d["connection_duration"] == 0.05
        assert d["send_duration"] == 0.1
        assert d["attempt_count"] == 1
        # Fields not set should still appear
        assert "receive_duration" in d


class TestStatsTrackerRecordP2POperation:
    def test_creates_activation_stats_and_populates_p2p(self):
        tracker = StatsTracker()
        op = P2POperationTimings(
            connection_duration=0.03,
            send_duration=0.07,
            receive_duration=0.2,
            total_start=500.0,
            total_end=500.3,
            total_duration=0.3,
            attempt_count=1,
            retry_count=0,
            total_backoff_time=0.0,
            bytes_sent=64,
            bytes_received=4096,
        )
        tracker.record_p2p_operation("act-1", op, direction="forward")

        stats = tracker.activation_stats["act-1"]
        assert stats.direction == "forward"
        p2p = stats.timing.p2p
        assert isinstance(p2p, P2PTimingDetail)
        assert p2p.connection_duration == 0.03
        assert p2p.send_duration == 0.07
        assert p2p.receive_duration == 0.2
        assert p2p.total_duration == 0.3
        assert p2p.attempt_count == 1
        assert p2p.bytes_received == 4096

    def test_updates_aggregate_p2p_bytes(self):
        tracker = StatsTracker()
        op = P2POperationTimings(bytes_received=2048)
        tracker.record_p2p_operation("act-2", op)
        assert tracker.p2p_bytes == 2048

        op2 = P2POperationTimings(bytes_received=1024)
        tracker.record_p2p_operation("act-3", op2)
        assert tracker.p2p_bytes == 3072

    def test_no_bytes_received_does_not_increment(self):
        tracker = StatsTracker()
        op = P2POperationTimings()  # bytes_received is None
        tracker.record_p2p_operation("act-4", op)
        assert tracker.p2p_bytes == 0

    def test_overwrites_p2p_on_existing_stats(self):
        tracker = StatsTracker()
        # Pre-create stats via ensure
        tracker.ensure_activation_stats("act-5", direction="backward")

        op = P2POperationTimings(send_duration=0.5, attempt_count=3, retry_count=2)
        tracker.record_p2p_operation("act-5", op)

        stats = tracker.activation_stats["act-5"]
        assert stats.direction == "backward"
        assert stats.timing.p2p.send_duration == 0.5
        assert stats.timing.p2p.attempt_count == 3

    def test_p2p_detail_round_trips_through_model_dump(self):
        """Ensure the full ActivationStats → JSON → dict round-trip works."""
        tracker = StatsTracker()
        op = P2POperationTimings(
            connection_duration=0.01,
            stream_open_duration=0.005,
            send_duration=0.02,
            receive_duration=0.1,
            total_duration=0.135,
            attempt_count=1,
            errors=[],
            bytes_sent=32,
            bytes_received=512,
        )
        tracker.record_p2p_operation("act-6", op, direction="forward")
        payload = tracker.get_activation_stats_payload("act-6")
        assert payload is not None
        p2p = payload["timing"]["p2p"]
        assert p2p["connection_duration"] == 0.01
        assert p2p["stream_open_duration"] == 0.005
        assert p2p["send_duration"] == 0.02
        assert p2p["receive_duration"] == 0.1
        assert p2p["bytes_sent"] == 32
        assert p2p["bytes_received"] == 512


class TestBackwardSubPhaseTimingFields:
    """Tests for granular backward sub-phase timing fields on ActivationTiming."""

    NEW_FIELDS = [
        "backward_gpu_setup",
        "backward_forward",
        "backward_loss",
        "backward_pass",
        "backward_grad_extract",
        "sample_download",
    ]

    def test_new_fields_exist_and_default_to_empty_stage(self):
        timing = ActivationTiming()
        for field_name in self.NEW_FIELDS:
            stage = getattr(timing, field_name)
            assert isinstance(stage, ActivationTimingStage)
            assert stage.start is None
            assert stage.end is None
            assert stage.duration is None

    def test_model_dump_includes_new_fields(self):
        timing = ActivationTiming()
        d = timing.model_dump()
        for field_name in self.NEW_FIELDS:
            assert field_name in d
            assert d[field_name]["start"] is None
            assert d[field_name]["end"] is None
            assert d[field_name]["duration"] is None

    def test_setting_duration_serialises_correctly(self):
        timing = ActivationTiming()
        timing.backward_forward.duration = 1.23
        timing.backward_loss.duration = 0.45
        timing.sample_download.start = 100.0
        timing.sample_download.end = 100.5
        timing.sample_download.duration = 0.5

        d = timing.model_dump()
        assert d["backward_forward"]["duration"] == 1.23
        assert d["backward_loss"]["duration"] == 0.45
        assert d["sample_download"]["start"] == 100.0
        assert d["sample_download"]["end"] == 100.5
        assert d["sample_download"]["duration"] == 0.5

    def test_round_trip_through_get_activation_stats_payload(self):
        tracker = StatsTracker()
        stats = tracker.ensure_activation_stats("act-bwd-1", direction="backward")
        stats.timing.backward_gpu_setup.duration = 0.1
        stats.timing.backward_forward.duration = 0.2
        stats.timing.backward_loss.duration = 0.3
        stats.timing.backward_pass.duration = 0.4
        stats.timing.backward_grad_extract.duration = 0.05
        stats.timing.sample_download.start = 50.0
        stats.timing.sample_download.end = 50.6
        stats.timing.sample_download.duration = 0.6

        payload = tracker.get_activation_stats_payload("act-bwd-1")
        assert payload is not None
        t = payload["timing"]
        assert t["backward_gpu_setup"]["duration"] == 0.1
        assert t["backward_forward"]["duration"] == 0.2
        assert t["backward_loss"]["duration"] == 0.3
        assert t["backward_pass"]["duration"] == 0.4
        assert t["backward_grad_extract"]["duration"] == 0.05
        assert t["sample_download"]["start"] == 50.0
        assert t["sample_download"]["end"] == 50.6
        assert t["sample_download"]["duration"] == 0.6

    def test_existing_fields_unaffected(self):
        """Adding new fields does not change defaults of existing fields."""
        timing = ActivationTiming()
        assert timing.queue.duration is None
        assert timing.download.duration is None
        assert timing.forward.duration is None
        assert timing.backward.duration is None
        assert isinstance(timing.p2p, P2PTimingDetail)
        assert timing.publish.duration is None
