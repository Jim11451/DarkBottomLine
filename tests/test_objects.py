"""
Unit tests for objects module.
"""

import pytest
import awkward as ak
import numpy as np
from darkbottomline.objects import (
    select_muons, select_electrons, select_taus, select_jets, select_fatjets,
    clean_jets_from_leptons, get_bjet_mask, build_objects
)


class TestObjectSelection:
    """Test object selection functions."""

    def test_select_muons(self):
        """Test muon selection."""
        # Create mock events
        events = ak.Array({
            "Muon": {
                "pt": [[25.0, 15.0, 30.0], [20.0, 10.0], [35.0]],
                "eta": [[1.0, 2.0, 0.5], [1.5, 3.0], [0.8]],
                "tightId": [[1, 0, 1], [1, 0], [1]],
                "pfIsoId": [[3, 2, 3], [3, 1], [3]]
            }
        })

        config = {
            "pt_min": 20.0,
            "eta_max": 2.4,
            "id": "tight",
            "iso": "tight"
        }

        mask = select_muons(events, config)

        # Check that high pt muons are selected
        assert ak.sum(mask[0]) == 2  # Two muons pass cuts in first event
        assert ak.sum(mask[1]) == 1  # One muon passes cuts in second event
        assert ak.sum(mask[2]) == 1  # One muon passes cuts in third event

    def test_select_electrons(self):
        """Test electron selection."""
        # Create mock events
        events = ak.Array({
            "Electron": {
                "pt": [[25.0, 15.0, 30.0], [20.0, 10.0], [35.0]],
                "eta": [[1.0, 2.0, 0.5], [1.5, 3.0], [0.8]],
                "cutBased": [[4, 2, 4], [4, 1], [4]],
                "pfIsoId": [[3, 2, 3], [3, 1], [3]]
            }
        })

        config = {
            "pt_min": 20.0,
            "eta_max": 2.5,
            "id": "tight",
            "iso": "tight"
        }

        mask = select_electrons(events, config)

        # Check that high pt electrons are selected
        assert ak.sum(mask[0]) == 2  # Two electrons pass cuts in first event
        assert ak.sum(mask[1]) == 1  # One electron passes cuts in second event
        assert ak.sum(mask[2]) == 1  # One electron passes cuts in third event

    def test_select_taus(self):
        """Test tau selection."""
        # Create mock events
        events = ak.Array({
            "Tau": {
                "pt": [[25.0, 15.0, 30.0], [20.0, 10.0], [35.0]],
                "eta": [[1.0, 2.0, 0.5], [1.5, 3.0], [0.8]],
                "idDeepTau2017v2p1VSjet": [[16, 8, 16], [16, 4], [16]],
                "decayMode": [[0, 1, 2], [0, 10], [1]]
            }
        })

        config = {
            "pt_min": 20.0,
            "eta_max": 2.3,
            "id": "tight",
            "decay_modes": [0, 1, 2, 10, 11]
        }

        mask = select_taus(events, config)

        # Check that high pt taus are selected
        assert ak.sum(mask[0]) == 2  # Two taus pass cuts in first event
        assert ak.sum(mask[1]) == 1  # One tau passes cuts in second event
        assert ak.sum(mask[2]) == 1  # One tau passes cuts in third event

    def test_select_jets(self):
        """Test jet selection."""
        # Create mock events
        events = ak.Array({
            "Jet": {
                "pt": [[35.0, 25.0, 40.0], [30.0, 20.0], [45.0]],
                "eta": [[1.0, 2.0, 0.5], [1.5, 3.0], [0.8]],
                "jetId": [[2, 1, 2], [2, 0], [2]]
            }
        })

        config = {
            "pt_min": 30.0,
            "eta_max": 2.4,
            "id": "tight"
        }

        mask = select_jets(events, config)

        # Check that high pt jets are selected
        assert ak.sum(mask[0]) == 2  # Two jets pass cuts in first event
        assert ak.sum(mask[1]) == 1  # One jet passes cuts in second event
        assert ak.sum(mask[2]) == 1  # One jet passes cuts in third event

    def test_select_fatjets(self):
        """Test fat jet selection."""
        # Create mock events
        events = ak.Array({
            "FatJet": {
                "pt": [[250.0, 150.0, 300.0], [200.0, 100.0], [350.0]],
                "eta": [[1.0, 2.0, 0.5], [1.5, 3.0], [0.8]],
                "jetId": [[2, 1, 2], [2, 0], [2]]
            }
        })

        config = {
            "pt_min": 200.0,
            "eta_max": 2.4,
            "id": "tight"
        }

        mask = select_fatjets(events, config)

        # Check that high pt fat jets are selected
        assert ak.sum(mask[0]) == 2  # Two fat jets pass cuts in first event
        assert ak.sum(mask[1]) == 1  # One fat jet passes cuts in second event
        assert ak.sum(mask[2]) == 1  # One fat jet passes cuts in third event

    def test_clean_jets_from_leptons(self):
        """Test jet cleaning from leptons."""
        # Create mock jets and leptons
        jets = ak.Array({
            "pt": [30.0, 35.0, 40.0],
            "eta": [1.0, 1.5, 2.0],
            "phi": [0.0, 1.0, 2.0]
        })

        leptons = ak.Array({
            "pt": [25.0, 30.0],
            "eta": [1.1, 1.6],
            "phi": [0.1, 1.1]
        })

        # Test cleaning (simplified - actual implementation would calculate Delta-R)
        mask = clean_jets_from_leptons(jets, leptons, dr_min=0.4)

        # Should return boolean mask
        assert len(mask) == len(jets)
        assert all(isinstance(x, bool) for x in mask)

    def test_get_bjet_mask(self):
        """Test b-jet tagging."""
        # Create mock jets
        jets = ak.Array({
            "pt": [30.0, 35.0, 40.0],
            "eta": [1.0, 1.5, 2.0],
            "btagDeepFlavB": [0.1, 0.5, 0.8],
            "btagDeepB": [0.05, 0.3, 0.7]
        })

        # Test medium working point
        config = {"algorithm": "deepJet", "wp": "medium"}
        mask = get_bjet_mask(jets, config)

        # Should return boolean mask
        assert len(mask) == len(jets)
        assert all(isinstance(x, bool) for x in mask)

    def test_build_objects(self):
        """Test complete object building."""
        # Create mock events
        events = ak.Array({
            "Muon": {
                "pt": [[25.0, 15.0], [20.0], [35.0]],
                "eta": [[1.0, 2.0], [1.5], [0.8]],
                "phi": [[0.0, 1.0], [0.5], [1.5]],
                "tightId": [[1, 0], [1], [1]],
                "pfIsoId": [[3, 2], [3], [3]]
            },
            "Electron": {
                "pt": [[30.0, 20.0], [25.0], [40.0]],
                "eta": [[1.2, 2.1], [1.8], [0.9]],
                "phi": [[0.2, 1.2], [0.7], [1.7]],
                "cutBased": [[4, 2], [4], [4]],
                "pfIsoId": [[3, 2], [3], [3]]
            },
            "Tau": {
                "pt": [[35.0, 25.0], [30.0], [45.0]],
                "eta": [[1.1, 2.1], [1.6], [0.9]],
                "phi": [[0.1, 1.1], [0.6], [1.6]],
                "idDeepTau2017v2p1VSjet": [[16, 8], [16], [16]],
                "decayMode": [[0, 1], [0], [1]]
            },
            "Jet": {
                "pt": [[40.0, 30.0], [35.0], [50.0]],
                "eta": [[1.0, 2.0], [1.5], [0.8]],
                "phi": [[0.0, 1.0], [0.5], [1.5]],
                "jetId": [[2, 1], [2], [2]],
                "btagDeepFlavB": [[0.1, 0.5], [0.3], [0.8]],
                "hadronFlavour": [[5, 0], [5], [5]]
            },
            "FatJet": {
                "pt": [[250.0, 150.0], [200.0], [300.0]],
                "eta": [[1.0, 2.0], [1.5], [0.8]],
                "phi": [[0.0, 1.0], [0.5], [1.5]],
                "jetId": [[2, 1], [2], [2]]
            }
        })

        config = {
            "objects": {
                "muons": {"pt_min": 20.0, "eta_max": 2.4, "id": "tight", "iso": "tight"},
                "electrons": {"pt_min": 20.0, "eta_max": 2.5, "id": "tight", "iso": "tight"},
                "taus": {"pt_min": 20.0, "eta_max": 2.3, "id": "tight", "decay_modes": [0, 1, 2, 10, 11]},
                "jets": {"pt_min": 30.0, "eta_max": 2.4, "id": "tight"},
                "fatjets": {"pt_min": 200.0, "eta_max": 2.4, "id": "tight"}
            },
            "btagging": {"algorithm": "deepJet", "wp": "medium"},
            "cleaning": {"dr_muon_jet": 0.4, "dr_electron_jet": 0.4, "dr_tau_jet": 0.4}
        }

        objects = build_objects(events, config)

        # Check that objects are built
        assert "muons" in objects
        assert "electrons" in objects
        assert "taus" in objects
        assert "jets" in objects
        assert "fatjets" in objects
        assert "bjets" in objects

        # Check that masks are created
        assert "muon_mask" in objects
        assert "electron_mask" in objects
        assert "tau_mask" in objects
        assert "jet_mask" in objects
        assert "fatjet_mask" in objects
        assert "bjet_mask" in objects


if __name__ == "__main__":
    pytest.main([__file__])
