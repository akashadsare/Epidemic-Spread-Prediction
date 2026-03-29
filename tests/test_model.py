import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import unittest
from models.seir_lstm import SEIR_LSTM
from config import MODEL_CONFIG, FEATURES


class TestSEIRLSTM(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = SEIR_LSTM(
            input_dim=MODEL_CONFIG.input_dim,
            hidden_dim=MODEL_CONFIG.hidden_dim,
            num_layers=MODEL_CONFIG.num_layers,
            dropout=MODEL_CONFIG.dropout,
            num_heads=4,
        ).to(self.device)

    def test_forward_shape(self):
        """Test that forward pass returns correct shapes."""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)

        N = torch.tensor([1e6, 2e6])
        S = torch.tensor([9e5, 1.8e6])
        E = torch.tensor([5e4, 1e5])
        I = torch.tensor([5e4, 1e5])
        R = torch.tensor([0.0, 0.0])

        preds, params = self.model(x, (S, E, I, R), N)

        self.assertEqual(preds.shape, (batch_size, seq_len))
        self.assertEqual(params[0].shape, (batch_size, seq_len))  # beta
        self.assertEqual(params[1].shape, (batch_size, seq_len))  # sigma
        self.assertEqual(params[2].shape, (batch_size, seq_len))  # gamma

    def test_seir_non_negative(self):
        """Test that predictions are non-negative."""
        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)

        N = torch.tensor([1e6])
        S_0, E_0, I_0, R_0 = 9e5, 5e4, 5e4, 0.0

        preds, _ = self.model(
            x,
            (
                torch.tensor([S_0]),
                torch.tensor([E_0]),
                torch.tensor([I_0]),
                torch.tensor([R_0]),
            ),
            N,
        )
        self.assertTrue(torch.all(preds >= 0))

    def test_with_vaccination(self):
        """Test forward pass with vaccination rates."""
        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)
        v = torch.full((batch_size, seq_len), 0.001)

        N = torch.tensor([1e6])
        S, E, I, R = (
            torch.tensor([9e5]),
            torch.tensor([5e4]),
            torch.tensor([5e4]),
            torch.tensor([0.0]),
        )

        preds, params = self.model(x, (S, E, I, R), N, v)
        self.assertEqual(preds.shape, (batch_size, seq_len))
        self.assertTrue(torch.all(preds >= 0))

    def test_invalid_input_dim_raises(self):
        """Test that wrong input dimension raises error."""
        with self.assertRaises((ValueError, RuntimeError)):
            x = torch.randn(1, 10, 5)  # Wrong feature dimension
            N = torch.tensor([1e6])
            S, E, I, R = (
                torch.tensor([9e5]),
                torch.tensor([5e4]),
                torch.tensor([5e4]),
                torch.tensor([0.0]),
            )
            self.model(x, (S, E, I, R), N)

    def test_nan_input_raises(self):
        """Test that NaN input raises ValueError."""
        with self.assertRaises(ValueError):
            x = torch.full((1, 10, MODEL_CONFIG.input_dim), float("nan"))
            N = torch.tensor([1e6])
            S, E, I, R = (
                torch.tensor([9e5]),
                torch.tensor([5e4]),
                torch.tensor([5e4]),
                torch.tensor([0.0]),
            )
            self.model(x, (S, E, I, R), N)

    def test_model_save_load(self):
        """Test model save and load with metadata."""
        config = MODEL_CONFIG.to_dict()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self.model.save_with_metadata(path, config)
            loaded_model, loaded_config = SEIR_LSTM.load_with_metadata(path, self.device)

            self.assertEqual(loaded_model.input_dim, self.model.input_dim)
            self.assertEqual(loaded_model.hidden_dim, self.model.hidden_dim)
            self.assertEqual(loaded_config["input_dim"], config["input_dim"])
        finally:
            os.unlink(path)

    def test_model_output_finite(self):
        """Test that model outputs are finite (no NaN or Inf)."""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)
        v = torch.full((batch_size, seq_len), 0.001)

        N = torch.tensor([1e6, 2e6])
        S = torch.tensor([9e5, 1.8e6])
        E = torch.tensor([5e4, 1e5])
        I = torch.tensor([5e4, 1e5])
        R = torch.tensor([0.0, 0.0])

        preds, params = self.model(x, (S, E, I, R), N, v)
        self.assertTrue(torch.all(torch.isfinite(preds)))
        self.assertTrue(torch.all(torch.isfinite(params[0])))
        self.assertTrue(torch.all(torch.isfinite(params[1])))
        self.assertTrue(torch.all(torch.isfinite(params[2])))

    def test_get_seir_compartments(self):
        """Test SEIR compartment extraction returns all expected keys."""
        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)
        N = torch.tensor([1e6])
        S, E, I, R = (
            torch.tensor([9e5]),
            torch.tensor([5e4]),
            torch.tensor([5e4]),
            torch.tensor([0.0]),
        )

        with torch.no_grad():
            result = self.model.get_seir_compartments(x, (S, E, I, R), N)

        for key in ["S", "E", "I", "R", "new_cases", "beta", "sigma", "gamma", "rt"]:
            self.assertIn(key, result)
            self.assertEqual(result[key].shape, (batch_size, seq_len))

    def test_population_conservation(self):
        """Test that S+E+I+R approximately equals N after simulation."""
        batch_size = 1
        seq_len = 10
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)
        N = torch.tensor([1e6])
        S, E, I, R = (
            torch.tensor([9e5]),
            torch.tensor([5e4]),
            torch.tensor([4e4]),
            torch.tensor([1e4]),
        )

        with torch.no_grad():
            result = self.model.get_seir_compartments(x, (S, E, I, R), N)

        for t in range(seq_len):
            total = result["S"][0, t] + result["E"][0, t] + result["I"][0, t] + result["R"][0, t]
            self.assertAlmostEqual(total.item(), N.item(), delta=N.item() * 0.01)

    def test_seir_params_bounds(self):
        """Test that predicted SEIR parameters are within physical bounds."""
        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, MODEL_CONFIG.input_dim)
        N = torch.tensor([1e6])
        S, E, I, R = (
            torch.tensor([9e5]),
            torch.tensor([5e4]),
            torch.tensor([5e4]),
            torch.tensor([0.0]),
        )

        with torch.no_grad():
            result = self.model.get_seir_compartments(x, (S, E, I, R), N)

        # Beta in [0, 2.5]
        self.assertTrue(torch.all(result["beta"] >= 0))
        self.assertTrue(torch.all(result["beta"] <= 2.5))
        # Sigma in [0.1, 0.5]
        self.assertTrue(torch.all(result["sigma"] >= 0.05))
        self.assertTrue(torch.all(result["sigma"] <= 0.55))
        # Gamma in [0.03, 0.35]
        self.assertTrue(torch.all(result["gamma"] >= 0.03))
        self.assertTrue(torch.all(result["gamma"] <= 0.35))


class TestLegacyLoading(unittest.TestCase):
    """Test backward compatibility with old checkpoint format."""

    def test_legacy_model_construction(self):
        """Test that legacy mode model can be constructed and forward pass works."""
        model = SEIR_LSTM(
            input_dim=11,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            legacy_mode=True,
        )
        x = torch.randn(1, 10, 11)
        N = torch.tensor([1e6])
        S, E, I, R = (
            torch.tensor([9e5]),
            torch.tensor([5e4]),
            torch.tensor([5e3]),
            torch.tensor([0.0]),
        )

        preds, params = model(x, (S, E, I, R), N)
        self.assertEqual(preds.shape, (1, 10))
        self.assertTrue(torch.all(torch.isfinite(preds)))

    def test_legacy_save_load(self):
        """Test saving and loading a legacy model preserves legacy_mode."""
        model = SEIR_LSTM(
            input_dim=11,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            legacy_mode=True,
        )
        config = {"input_dim": 11, "hidden_dim": 64}
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            model.save_with_metadata(path, config)
            loaded, loaded_config = SEIR_LSTM.load_with_metadata(path, torch.device("cpu"))
            self.assertTrue(loaded.legacy_mode)
            self.assertEqual(loaded.input_dim, 11)
        finally:
            os.unlink(path)


class TestConfig(unittest.TestCase):
    def test_model_config_defaults(self):
        """Test ModelConfig has correct defaults."""
        config = MODEL_CONFIG
        self.assertEqual(config.input_dim, len(FEATURES))
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.dropout, 0.25)

    def test_model_config_to_dict(self):
        """Test ModelConfig.to_dict() returns proper dict."""
        config_dict = MODEL_CONFIG.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("input_dim", config_dict)
        self.assertIn("hidden_dim", config_dict)

    def test_features_count(self):
        """Test that FEATURES has the expected number of features."""
        # 11 core + 8 log_diff + 6 environmental = 25
        self.assertEqual(len(FEATURES), 25)


if __name__ == "__main__":
    unittest.main()
