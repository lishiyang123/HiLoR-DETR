import unittest

from _bootstrap import ROOT  # noqa: F401
import torch

from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer


def make_decoder(return_features=False, ufd=False):
    return RTDETRTransformer(
        num_classes=3,
        hidden_dim=32,
        num_queries=5,
        feat_channels=[8, 16, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=2,
        nhead=4,
        num_decoder_layers=1,
        dim_feedforward=64,
        num_denoising=0,
        eval_spatial_size=None,
        hilor_return_decoder_features=return_features,
        ufd_enabled=ufd,
    )


class DecoderContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.features = [
            torch.randn(1, 8, 4, 4),
            torch.randn(1, 16, 2, 2),
            torch.randn(1, 32, 1, 1),
        ]

    def test_feature_exposure_does_not_change_native_predictions(self):
        native = make_decoder(False, False).eval()
        capture = make_decoder(True, False).eval()
        capture.load_state_dict(native.state_dict(), strict=True)
        with torch.no_grad():
            native_output = native(self.features)
            capture_output = capture(self.features)
        self.assertEqual(set(native_output), {"pred_logits", "pred_boxes"})
        self.assertIn("hilor_decoder_features", capture_output)
        self.assertTrue(torch.equal(native_output["pred_logits"], capture_output["pred_logits"]))
        self.assertTrue(torch.equal(native_output["pred_boxes"], capture_output["pred_boxes"]))
        self.assertEqual(tuple(capture_output["hilor_decoder_features"].shape), (1, 5, 32))

    def test_ufd_head_is_training_only(self):
        decoder = make_decoder(False, True)
        decoder.train()
        training_features = [feature.repeat(2, 1, 1, 1) for feature in self.features]
        output = decoder(training_features)
        self.assertEqual(tuple(output["ufd_logits"].shape), (2, 21, 132))
        self.assertEqual(tuple(output["ufd_ref_points"].shape), (21, 4))
        decoder.eval()
        with torch.no_grad():
            output = decoder(self.features)
        self.assertNotIn("ufd_logits", output)

    def test_deploy_strip_preserves_predictions_and_removes_ufd_state(self):
        decoder = make_decoder(False, True).eval()
        with torch.no_grad():
            before = decoder(self.features)
        self.assertTrue(any(key.startswith("ufd_head.") for key in decoder.state_dict()))
        decoder.convert_to_deploy()
        self.assertFalse(any(key.startswith("ufd_") for key in decoder.state_dict()))
        with torch.no_grad():
            after = decoder(self.features)
        self.assertTrue(torch.equal(before["pred_logits"], after["pred_logits"]))
        self.assertTrue(torch.equal(before["pred_boxes"], after["pred_boxes"]))

    def test_ufd_toggle_preserves_shared_initialization_and_rng_stream(self):
        torch.manual_seed(73)
        native = make_decoder(False, False)
        native_next_random = torch.rand(8)
        torch.manual_seed(73)
        with_ufd = make_decoder(False, True)
        ufd_next_random = torch.rand(8)

        native_state = native.state_dict()
        ufd_state = with_ufd.state_dict()
        for key, value in native_state.items():
            self.assertIn(key, ufd_state)
            self.assertTrue(torch.equal(value, ufd_state[key]), key)
        self.assertTrue(torch.equal(native_next_random, ufd_next_random))
        self.assertTrue(
            torch.equal(with_ufd.ufd_head.bias, torch.zeros_like(with_ufd.ufd_head.bias))
        )


if __name__ == "__main__":
    unittest.main()
