"""
Test script to verify plot exclusions work correctly.
"""

import unittest
from darkbottomline.plotting import PlotManager


class TestPlotExclusions(unittest.TestCase):
    """Test plot exclusion logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.plot_manager = PlotManager()

    def test_1b_sr_exclusions(self):
        """Test that 1b SR excludes jet3 and lepton variables."""
        region = "1b:SR"
        excluded = self.plot_manager._get_excluded_variables_for_region(region)

        # Should exclude jet3 variables
        self.assertIn("jet3_pt", excluded)
        self.assertIn("m_jet1jet3", excluded)

        # Should exclude lepton variables
        self.assertIn("lep1_pt", excluded)
        self.assertIn("lep2_pt", excluded)
        self.assertIn("n_muons", excluded)
        self.assertIn("n_electrons", excluded)

        # Should NOT exclude jet1 or jet2
        self.assertNotIn("jet1_pt", excluded)
        self.assertNotIn("jet2_pt", excluded)

    def test_2b_sr_exclusions(self):
        """Test that 2b SR excludes lepton variables but keeps jet3."""
        region = "2b:SR"
        excluded = self.plot_manager._get_excluded_variables_for_region(region)

        # Should exclude lepton variables
        self.assertIn("lep1_pt", excluded)
        self.assertIn("n_muons", excluded)

        # Should NOT exclude jet3 variables (2b SR has 3 jets)
        self.assertNotIn("jet3_pt", excluded)
        self.assertNotIn("m_jet1jet3", excluded)

    def test_top_cr_exclusions(self):
        """Test that Top CRs exclude z_mass and z_pt."""
        regions = ["1b:CR_Top_mu", "2b:CR_Top_el", "2b:CR_Top_mu"]

        for region in regions:
            excluded = self.plot_manager._get_excluded_variables_for_region(region)

            # Should exclude z_mass and z_pt
            self.assertIn("z_mass", excluded, f"z_mass should be excluded from {region}")
            self.assertIn("z_pt", excluded, f"z_pt should be excluded from {region}")

            # Should NOT exclude lepton variables (CRs have leptons)
            self.assertNotIn("lep1_pt", excluded)
            self.assertNotIn("lep2_pt", excluded)

    def test_w_cr_exclusions(self):
        """Test that W CRs exclude z_mass and z_pt."""
        regions = ["1b:CR_Wlnu_mu", "1b:CR_Wlnu_el"]

        for region in regions:
            excluded = self.plot_manager._get_excluded_variables_for_region(region)

            # Should exclude z_mass and z_pt
            self.assertIn("z_mass", excluded, f"z_mass should be excluded from {region}")
            self.assertIn("z_pt", excluded, f"z_pt should be excluded from {region}")

            # Should NOT exclude lepton variables (CRs have leptons)
            self.assertNotIn("lep1_pt", excluded)

    def test_z_cr_inclusions(self):
        """Test that Z CRs include z_mass and z_pt (not excluded)."""
        regions = ["1b:CR_Zll_mu", "1b:CR_Zll_el", "2b:CR_Zll_mu", "2b:CR_Zll_el"]

        for region in regions:
            excluded = self.plot_manager._get_excluded_variables_for_region(region)

            # Should NOT exclude z_mass and z_pt (Z CRs need these plots)
            self.assertNotIn("z_mass", excluded, f"z_mass should NOT be excluded from {region}")
            self.assertNotIn("z_pt", excluded, f"z_pt should NOT be excluded from {region}")

    def test_1b_cr_jet3_exclusions(self):
        """Test that 1b CRs exclude jet3 variables."""
        regions = ["1b:CR_Wlnu_mu", "1b:CR_Zll_mu"]

        for region in regions:
            excluded = self.plot_manager._get_excluded_variables_for_region(region)

            # Should exclude jet3 variables (1b regions have <=2 jets)
            self.assertIn("jet3_pt", excluded, f"jet3_pt should be excluded from {region}")
            self.assertIn("m_jet1jet3", excluded, f"m_jet1jet3 should be excluded from {region}")

    def test_2b_cr_jet3_inclusions(self):
        """Test that 2b CRs include jet3 variables (Top CR may have >3 jets)."""
        region = "2b:CR_Top_mu"
        excluded = self.plot_manager._get_excluded_variables_for_region(region)

        # Should NOT exclude jet3 variables (Top CR may have >3 jets)
        self.assertNotIn("jet3_pt", excluded)
        self.assertNotIn("m_jet1jet3", excluded)

    def test_custom_exclusions(self):
        """Test custom exclusions from config."""
        custom_config = {
            "region_exclusions": {
                "1b:SR": ["custom_var1", "custom_var2"],
                "Top": ["custom_var3"],
            }
        }
        plot_manager = PlotManager(config=custom_config)

        # Test exact region match
        excluded = plot_manager._get_excluded_variables_for_region("1b:SR")
        self.assertIn("custom_var1", excluded)
        self.assertIn("custom_var2", excluded)

        # Test pattern match
        excluded = plot_manager._get_excluded_variables_for_region("2b:CR_Top_mu")
        self.assertIn("custom_var3", excluded)
        self.assertIn("z_mass", excluded)  # Should still have default exclusion


if __name__ == "__main__":
    unittest.main()



